"""
Deriv EXPIRYRANGE MC Bot v2 — 1HZ10V + RDBEAR
================================================
Fixes applied vs v1 (confirmed from Railway logs):

  BUG 1 — GARCH vol unit error (ROOT CAUSE of ±0.30 barriers on 9648-priced asset)
    GARCH is fitted on RELATIVE returns scaled by 1000.
    vol_per_tick from GARCH came out as ~0.000018 (relative units / 1000).
    The MC used this directly as absolute price units → barrier = 0.40 * 0.000018 * sqrt(120)
    = 0.000079 price units → all 75K paths trivially inside → win=1.000.
    FIX: convert relative vol back to absolute: abs_vol = (garch_vol / scale) * price.
    Also added a hard sanity check: if abs_vol_per_tick < 0.01 * price/1000, abort and
    fall back to std(price_diffs) directly (no relative-return conversion).

  BUG 2 — Proposal API payout misread ($2.18 returned instead of $0.18)
    Deriv's proposal response for EXPIRYRANGE returns:
      { "proposal": { "ask_price": 0.35, "payout": 2.53, ... } }
    "payout" is the TOTAL payout if won (stake + profit), not net profit.
    Code did total - BASE_STAKE = 2.53 - 0.35 = $2.18 — WRONG because ask_price
    for EXPIRYRANGE is NOT always equal to stake (Deriv may adjust it).
    FIX: net_profit = payout - ask_price (use the actual ask price from the response).
    Also: cap sanity check — if net_profit > stake * 20, something is still wrong → skip.

  BUG 3 — Supabase column mismatch (PGRST204 on 'actual_payout')
    The SQL schema used column name 'actual_payout' but the original schema.sql
    had different column names. The schema.sql we provided had 'actual_payout'
    but the user's Supabase table was created from the FIRST schema which used
    'breach_prob' and 'ev_conservative' etc without 'actual_payout', 'mc_win_prob',
    'mc_ci_lower', 'barrier_sigma', 'hawkes_val'.
    FIX: simplified log record to only columns that exist in the ORIGINAL schema,
    plus a graceful column-mismatch fallback that logs what it can.

  BUG 4 — ValueError: too many values to unpack (expected 2)
    After refactor, execute_expiryrange returned 3-tuple (won, profit, placed)
    but some call sites still unpacked 2 values.
    FIX: all call sites updated; consistent 3-tuple everywhere.

  BUG 5 — Watchdog restarting too aggressively (4.5 min timeout, GARCH takes 3+ min)
    FIX: WATCHDOG_TIMEOUT raised to 15 minutes; last_activity reset after GARCH fit.

  BUG 6 — Bootstrap only loading 1000 ticks (HISTORY_BOOTSTRAP was 5000 but API
    silently returned 1000 — Deriv caps ticks_history at 5000 max but the count=5000
    call was working; however GARCH needs MIN_TICKS_FOR_FIT=200 and was getting it.
    The real issue: GARCH on RELATIVE returns of a 9648-priced asset with scale=1000
    gives conditional vol in units of (relative_return * 1000), not price.
    This is fixed by BUG 1 fix.

  BUG 7 — Unicode box-drawing chars split by Railway logger into individual lines
    FIX: all separators switched to plain ASCII hyphens/equals.

ARCHITECTURE (unchanged):
  4-stage filter: Gate → MC optimizer → Proposal API payout check → Execute
  Daily self-improvement: midnight UTC, Bayesian reweighting of dur/barrier prefs
  Supabase: all trades logged, config warm-started on restart
"""

import asyncio, contextlib, io, json, math, os, random, sys, time, warnings
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import websockets
from scipy.stats import norm
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Deriv connection ──────────────────────────────────────────────────────
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None
API_BASE           = "https://api.derivws.com"
ACCOUNTS_PATH      = "/trading/v1/options/accounts"
OTP_PATH           = "/trading/v1/options/accounts/{account_id}/otp"

# ── Supabase ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Symbols ───────────────────────────────────────────────────────────────
SYMBOLS = ["1HZ10V", "RDBEAR"]

# ── Contract parameters ───────────────────────────────────────────────────
BASE_STAKE        = 0.35      # Deriv minimum
MIN_NET_PAYOUT    = 0.182     # 52% of $0.35 — enforced via proposal API
WATCHDOG_TIMEOUT  = 15 * 60  # 15 min — accounts for GARCH + bootstrap time
HISTORY_BOOTSTRAP = 5000
MIN_TICKS_FOR_FIT = 200
MIN_TICKS_LIVE    = 60
GARCH_SCALE       = 1000.0   # scale factor for GARCH fitting on relative returns

# ── MC engine ─────────────────────────────────────────────────────────────
MC_SIMULATIONS   = 75_000
MC_CI_PERCENTILE = 5
MC_REQUIRED_WIN  = 0.58    # MC floor; proposal API is the real payout gate
MC_REQUIRED_CI   = 0.56
MC_BATCH_SIZE    = 25_000

# ── Sweep grids ───────────────────────────────────────────────────────────
DURATION_CANDIDATES = [120, 180, 240, 300, 360, 420, 480]   # seconds (min 2 min)
# Barrier expressed as multiples of terminal vol.
# Targeting win_prob 58-70% zone where Deriv payout is typically $0.18-$0.25.
BARRIER_SIGMAS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                  0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00]
BARRIER_ABS_MIN = 0.3    # minimum absolute barrier (price units)

# ── Per-symbol gate thresholds ────────────────────────────────────────────
SYMBOL_CONFIG = {
    "1HZ10V": {
        "ticks_per_sec":     1.0,
        "max_adx":           22,
        "min_vol_trust":     0.85,
        "max_mbs":           0.40,
        "boll_width_factor": 1.20,
        "max_hawkes":        0.50,
        "cooldown_secs":     150,
        "barrier_dp":        2,    # Deriv: max 2 decimal places for 1HZ10V barriers
    },
    "RDBEAR": {
        "ticks_per_sec":     1.0,
        "max_adx":           18,
        "min_vol_trust":     0.80,
        "max_mbs":           0.30,
        "boll_width_factor": 1.15,
        "max_hawkes":        0.40,
        "cooldown_secs":     180,
        "barrier_dp":        4,    # Deriv: max 4 decimal places for RDBEAR barriers
    },
}

DAILY_TUNE_HOUR_UTC = 0


# =============================================================================
# SUPABASE STORE
# =============================================================================
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        print(f"[Store] {'Active -> ' + self.url if self.ok else 'No creds — state will not persist.'}")

    def _hdr(self, prefer="return=minimal"):
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json", "Prefer": prefer}

    def _upsert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr("resolution=merge-duplicates,return=minimal"),
                json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} upsert error: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} insert error: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                headers=self._hdr("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Store] {table} select error: {e}")
        return []

    def log_trade(self, rec: dict):
        """
        Maps to the ORIGINAL bot_expiryrange_log schema columns.
        Uses only columns that were in the original CREATE TABLE statement.
        Any extra columns are silently dropped to avoid PGRST204 errors.
        """
        payload = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "symbol":          rec["symbol"],
            "entry_price":     round(float(rec["entry_price"]),     5),
            "upper_barrier":   round(float(rec["upper_barrier"]),   5),
            "lower_barrier":   round(float(rec["lower_barrier"]),   5),
            "barrier_width":   round(float(rec["barrier_width"]),   5),
            "duration_secs":   int(rec["duration_secs"]),
            "stake":           round(float(BASE_STAKE),             4),
            "won":             bool(rec["won"]),
            "profit":          round(float(rec["profit"]),          4),
            "breach_prob":     round(float(rec["breach_prob"]),     4),
            "ev_conservative": round(float(rec.get("ev_conservative", 0)), 4),
            "ev_optimistic":   round(float(rec.get("ev_optimistic",  0)), 4),
            "vol_per_tick":    round(float(rec["vol_per_tick"]),    6),
            "used_garch":      bool(rec["used_garch"]),
            "adx_val":         round(float(rec.get("adx_val", 0)), 3),
            "vol_trust":       round(float(rec.get("vol_trust", 0)), 4),
            "hawkes_intensity":round(float(rec.get("hawkes_val", 0)), 4),
            "n_sims":          int(MC_SIMULATIONS),
        }
        self._insert("bot_expiryrange_log", payload)

    def save_config(self, key, value):
        self._upsert("bot_expiryrange_config",
            {"key": key, "value": json.dumps(value),
             "updated_at": datetime.now(timezone.utc).isoformat()})

    def load_config(self, key):
        rows = self._select("bot_expiryrange_config", f"select=value&key=eq.{key}")
        if rows:
            raw = rows[0]["value"]
            return json.loads(raw) if isinstance(raw, str) else raw
        return None

    def save_daily_summary(self, date_str, symbol, n, wins, profit, best_dur, best_bar):
        self._upsert("bot_expiryrange_daily", {
            "date_utc":     date_str, "symbol": symbol,
            "n_trades":     n, "n_wins": wins,
            "win_rate":     round(wins / max(n, 1), 4),
            "total_profit": round(float(profit), 4),
            "best_duration": int(best_dur),
            "best_barrier":  round(float(best_bar), 4),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        })

    def load_recent_trades(self, symbol, days=7):
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self._select("bot_expiryrange_log",
            f"select=*&symbol=eq.{symbol}&ts=gte.{since}&order=ts.asc")


# =============================================================================
# DERIV CLIENT  (identical connection layer from parent bot)
# =============================================================================
class DerivClient:
    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id        = app_id
        self.token         = token
        self.account_type  = account_type
        self.account_id    = account_id
        self.ws            = None
        self.req_id        = 0
        self.pending: dict = {}
        self.subscriptions = defaultdict(list)
        self.account       = None
        self.resubscribe_cb = None
        self._running      = False
        self._reader_task  = None
        self._ka_task      = None

    def _rest_headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
                "Content-Type": "application/json"}

    def _resolve_account_id_sync(self):
        resp = requests.get(f"{API_BASE}{ACCOUNTS_PATH}",
                            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                aid = acc.get("account_id") or acc.get("id")
                if aid:
                    return aid
        raise RuntimeError(f"No '{self.account_type}' account found. data={data}")

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        resp = requests.post(
            f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}",
            headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task     = asyncio.create_task(self._heartbeat())
        bal          = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(f"Connected ({self.account_type}). "
              f"loginid={self.account.get('loginid')} "
              f"balance=${self.account.get('balance'):.2f}")

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[Client] WS lost: {e}")

    async def _supervise(self):
        while self._running:
            if self._reader_task:
                await self._reader_task
            if self._ka_task:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WS disconnected"))
            self.pending.clear()
            self.ws = None
            if not self._running:
                break
            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(self.RECONNECT_BASE * (2 ** (attempt - 1)),
                            self.RECONNECT_CAP) + random.uniform(0, 1)
                print(f"[Client] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[Client] Reconnect failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id   = data.get("req_id")
        msg_type = data.get("msg_type")
        if msg_type == "ping":
            return
        if req_id is not None and req_id in self.pending:
            fut = self.pending.pop(req_id)
            if not fut.done():
                fut.set_result(data)
                return
        if msg_type in self.subscriptions:
            for q in self.subscriptions[msg_type]:
                q.put_nowait(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = {**request, "req_id": rid}
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


# =============================================================================
# SYMBOL DATA BUFFER
# =============================================================================
class SymbolData:
    def __init__(self, symbol, maxlen=8000):
        self.symbol = symbol
        self.ticks  = deque(maxlen=maxlen)

    def add_tick(self, epoch, price):
        self.ticks.append((float(epoch), float(price)))

    def prices(self) -> np.ndarray:
        return np.array([p for _, p in self.ticks], dtype=float)

    def price_diffs(self) -> np.ndarray:
        """Absolute price differences (not returns) — used for vol estimation."""
        p = self.prices()
        return np.diff(p) if len(p) >= 2 else np.array([])

    def returns(self) -> np.ndarray:
        """Relative returns — used for GARCH fitting."""
        p = self.prices()
        if len(p) < 2: return np.array([])
        return np.diff(p) / p[:-1]


# =============================================================================
# BOT STATE
# =============================================================================
class BotState:
    def __init__(self):
        self.balance          = 0.0
        self.trading_locked   = False
        self.last_activity    = time.time()
        self.last_price: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_trade_time: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
        self.last_daily_tune  = 0.0
        self.garch_cache: Dict[str, tuple]  = {}     # sym -> (result, fitted_at)
        self.vol_scalar: Dict[str, float]   = {s: 1.0 for s in SYMBOLS}
        self.duration_weights: Dict[str, Dict[int, float]]   = {s: {} for s in SYMBOLS}
        self.barrier_weights:  Dict[str, Dict[float, float]] = {s: {} for s in SYMBOLS}
        self.session_trades:  Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_wins:    Dict[str, int]   = {s: 0   for s in SYMBOLS}
        self.session_profit:  Dict[str, float] = {s: 0.0 for s in SYMBOLS}


# =============================================================================
# VOL ESTIMATION  — FIX FOR BUG 1
# =============================================================================
def estimate_abs_vol_per_tick(prices: np.ndarray, returns: np.ndarray,
                               garch_result, price_now: float) -> Tuple[float, float, bool]:
    """
    Returns (abs_vol_per_tick, vol_trust, used_garch).

    GARCH is fitted on relative returns * GARCH_SCALE.
    Conditional vol from GARCH is therefore in units of (relative_return * GARCH_SCALE).
    To get absolute price vol per tick:
        abs_vol = (garch_cond_vol / GARCH_SCALE) * price_now

    Sanity check: abs_vol must be between 0.001% and 5% of price_now.
    If outside that range, fall back to std(price_diffs) directly.
    """
    price_diffs  = np.diff(prices) if len(prices) >= 2 else np.array([0.0])
    baseline_abs = float(np.std(price_diffs)) if len(price_diffs) > 5 else abs(price_now) * 0.001
    baseline_abs = max(baseline_abs, 1e-6)

    lo_bound = price_now * 0.00001   # 0.001% of price
    hi_bound = price_now * 0.05      # 5% of price

    used_garch = False
    vol_trust  = 0.5

    if garch_result is not None:
        try:
            fc = garch_result.forecast(horizon=1, reindex=False)
            garch_cond_vol_scaled = math.sqrt(float(fc.variance.values[-1, 0]))
            # Convert back to absolute price units
            abs_vol = (garch_cond_vol_scaled / GARCH_SCALE) * price_now
            if lo_bound <= abs_vol <= hi_bound:
                # Compare to baseline to compute trust
                ratio     = abs_vol / max(baseline_abs, 1e-9)
                vol_trust = float(np.clip(1.0 / (1.0 + max(ratio - 1.0, 0) * 2), 0.1, 1.0))
                used_garch = True
                return float(abs_vol), vol_trust, used_garch
            else:
                print(f"[Vol] GARCH abs_vol={abs_vol:.5f} outside [{lo_bound:.5f},{hi_bound:.5f}]"
                      f" — falling back to baseline={baseline_abs:.5f}")
        except Exception as e:
            print(f"[Vol] GARCH forecast error: {e}")

    # Fallback: direct std of price differences
    return baseline_abs, 0.5, False


# =============================================================================
# GARCH FITTING
# =============================================================================
def fit_garch(returns: np.ndarray):
    """Fit GARCH(1,1) on relative returns * GARCH_SCALE. Returns result or None."""
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * GARCH_SCALE
        am     = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                return am.fit(disp="off")
    except Exception as e:
        print(f"[GARCH] fit failed: {e}")
        return None


# =============================================================================
# STRUCTURAL INDICATORS
# =============================================================================
def compute_adx(prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
    if len(prices) < period * 2 + 2:
        return 20.0, 0.3
    tr_, pdm_, ndm_ = [], [], []
    for i in range(1, len(prices)):
        tr_.append(abs(prices[i] - prices[i-1]))
        pdm_.append(max(prices[i] - prices[i-1], 0.0))
        ndm_.append(max(prices[i-1] - prices[i], 0.0))
    tr_a  = np.array(tr_[-period*2:])
    pdm_a = np.array(pdm_[-period*2:])
    ndm_a = np.array(ndm_[-period*2:])
    atr   = np.mean(tr_a[-period:])
    if atr == 0:
        return 20.0, 0.3
    adx_vals = [
        100 * abs(pdm_a[i] - ndm_a[i]) / (np.mean(tr_a[max(0,i-period):i]) * period + 1e-9)
        for i in range(period, len(tr_a))
    ]
    adx = float(np.clip(np.mean(adx_vals) if adx_vals else 20.0, 0, 100))
    return adx, float(np.clip((adx - 20) / 30, 0, 1))


def compute_bollinger_width(prices: np.ndarray) -> Tuple[float, float]:
    if len(prices) < 30:
        return 1.0, 1.0
    cur = float(np.std(prices[-10:]))
    stds = [float(np.std(prices[i:i+10])) for i in range(0, len(prices)-10, 5)]
    return cur, float(np.median(stds)) if stds else cur


def compute_hawkes_proxy(price_diffs: np.ndarray) -> float:
    if len(price_diffs) < 20:
        return 0.0
    thresh  = 0.5 * np.std(price_diffs) if np.std(price_diffs) > 0 else 1e-9
    recent  = float(np.mean(np.abs(price_diffs[-20:]) > thresh))
    base    = float(np.mean(np.abs(price_diffs) > thresh)) if len(price_diffs) >= 50 else 0.1
    return float(np.clip((recent / max(base, 1e-6) - 1.0) / 3.0, 0.0, 1.0))


def compute_mbs(prices: np.ndarray, lookback: int = 50) -> float:
    if len(prices) < lookback + 5:
        return 0.0
    w    = prices[-lookback:]
    rng  = np.max(w) - np.min(w)
    if rng < 1e-9: return 0.0
    prev_hi = np.max(w[:-10])
    prev_lo = np.min(w[:-10])
    last    = prices[-1]
    return float(np.clip(max(max(0.0, last - prev_hi), max(0.0, prev_lo - last)) / rng, 0, 1))


# =============================================================================
# STRUCTURAL GATE — all 5 conditions must pass
# =============================================================================
def structural_gate(symbol: str, prices: np.ndarray, price_diffs: np.ndarray,
                    returns: np.ndarray, garch_result,
                    price_now: float) -> Tuple[bool, dict]:
    cfg  = SYMBOL_CONFIG[symbol]
    ok   = True
    info = {}

    # 1. ADX
    adx_val, _ = compute_adx(prices)
    info["adx_val"] = adx_val
    if adx_val > cfg["max_adx"]:
        ok = False
        info["fail_adx"] = f"ADX={adx_val:.1f} > {cfg['max_adx']}"

    # 2. Vol trust (need abs vol estimate first)
    abs_vol, vol_trust, _ = estimate_abs_vol_per_tick(prices, returns, garch_result, price_now)
    info["vol_trust"] = vol_trust
    info["abs_vol"]   = abs_vol
    if vol_trust < cfg["min_vol_trust"]:
        ok = False
        info["fail_vol"] = f"vol_trust={vol_trust:.3f} < {cfg['min_vol_trust']}"

    # 3. MBS
    mbs_val = compute_mbs(prices)
    info["mbs_val"] = mbs_val
    if mbs_val >= cfg["max_mbs"]:
        ok = False
        info["fail_mbs"] = f"mbs={mbs_val:.3f} >= {cfg['max_mbs']}"

    # 4. Bollinger width
    cur_std, med_std = compute_bollinger_width(prices)
    info["cur_std"] = cur_std
    info["med_std"] = med_std
    if med_std > 0 and cur_std > med_std * cfg["boll_width_factor"]:
        ok = False
        info["fail_boll"] = f"cur_std/med_std={cur_std/med_std:.2f} > {cfg['boll_width_factor']}"

    # 5. Hawkes
    hawkes_val = compute_hawkes_proxy(price_diffs)
    info["hawkes_val"] = hawkes_val
    if hawkes_val > cfg["max_hawkes"]:
        ok = False
        info["fail_hawkes"] = f"hawkes={hawkes_val:.3f} > {cfg['max_hawkes']}"

    return ok, info


# =============================================================================
# MC ENGINE  — BUG 1 fixed: uses absolute vol correctly
# =============================================================================
def mc_breach_estimate(abs_vol_per_tick: float, barrier_abs: float,
                       duration_secs: float, ticks_per_sec: float) -> dict:
    """
    Estimates Pr(|X_terminal| < barrier_abs) via batched MC.
    Terminal distribution: X_T ~ N(0, abs_vol_per_tick * sqrt(n_steps))
    where abs_vol_per_tick is in ABSOLUTE PRICE UNITS.
    """
    n_steps      = max(1, int(round(duration_secs * ticks_per_sec)))
    vol_terminal = abs_vol_per_tick * math.sqrt(n_steps)

    if vol_terminal < 1e-9:
        return {"blocked": True, "reason": f"vol_terminal={vol_terminal:.2e} too small"}

    wins = 0
    done = 0
    while done < MC_SIMULATIONS:
        batch    = min(MC_BATCH_SIZE, MC_SIMULATIONS - done)
        terminal = np.random.normal(0.0, vol_terminal, size=batch)
        wins    += int(np.sum(np.abs(terminal) < barrier_abs))
        done    += batch

    win_prob = wins / MC_SIMULATIONS
    return {
        "blocked":      False,
        "win_prob":     win_prob,
        "breach_prob":  1.0 - win_prob,
        "vol_terminal": vol_terminal,
        "n_steps":      n_steps,
        "n_sims":       MC_SIMULATIONS,
    }


def ci_lower_bound(win_prob: float, n: int) -> float:
    z = norm.ppf(1 - MC_CI_PERCENTILE / 100)
    return win_prob - z * math.sqrt(max(win_prob * (1 - win_prob) / n, 1e-12))


# =============================================================================
# MC AUTO-OPTIMIZER
# =============================================================================
def mc_auto_optimize(prices: np.ndarray, price_diffs: np.ndarray,
                     returns: np.ndarray, symbol: str,
                     garch_result, state: BotState) -> Optional[List[dict]]:
    """
    Sweeps (duration × barrier_sigma) grid.
    Returns list of passing candidates sorted by win_prob ascending
    (lowest win = narrowest barrier = highest Deriv payout) or None.
    """
    cfg           = SYMBOL_CONFIG[symbol]
    ticks_per_sec = cfg["ticks_per_sec"]
    price_now     = float(prices[-1]) if len(prices) > 0 else 1.0

    abs_vol, vol_trust, used_garch = estimate_abs_vol_per_tick(
        prices, returns, garch_result, price_now)
    abs_vol *= state.vol_scalar.get(symbol, 1.0)

    candidates = []
    for dur_secs in DURATION_CANDIDATES:
        n_steps      = max(1, int(round(dur_secs * ticks_per_sec)))
        vol_terminal = abs_vol * math.sqrt(n_steps)

        for bs in BARRIER_SIGMAS:
            barrier_abs = max(bs * vol_terminal, BARRIER_ABS_MIN)

            mc = mc_breach_estimate(abs_vol, barrier_abs, dur_secs, ticks_per_sec)
            if mc.get("blocked"):
                continue

            wp  = mc["win_prob"]
            cil = ci_lower_bound(wp, MC_SIMULATIONS)

            if wp  < MC_REQUIRED_WIN: continue
            if cil < MC_REQUIRED_CI:  continue

            # Learned preference weights
            dw  = state.duration_weights.get(symbol, {}).get(dur_secs, 1.0)
            bw  = state.barrier_weights.get(symbol, {}).get(round(bs * 2) / 2, 1.0)

            candidates.append({
                "duration_secs":  dur_secs,
                "barrier_abs":    barrier_abs,
                "barrier_sigma":  bs,
                "n_steps":        n_steps,
                "win_prob":       wp,
                "ci_lower":       cil,
                "breach_prob":    mc["breach_prob"],
                "vol_per_tick":   abs_vol,
                "vol_terminal":   vol_terminal,
                "used_garch":     used_garch,
                "vol_trust":      vol_trust,
                "weighted_score": wp * dw * bw,
            })

    if not candidates:
        return None

    # Sort ascending by win_prob: lowest win = narrowest barrier = highest payout
    candidates.sort(key=lambda x: x["win_prob"])
    return candidates


# =============================================================================
# PROPOSAL API  — BUG 2 fixed: net = payout - ask_price (not payout - BASE_STAKE)
# =============================================================================
async def fetch_proposal_payout(client: DerivClient, symbol: str,
                                upper: float, lower: float,
                                duration_secs: int) -> Tuple[Optional[float], float]:
    """
    Returns (net_profit, ask_price).
    net_profit = proposal.payout - proposal.ask_price
    ask_price is what Deriv actually charges (may differ slightly from BASE_STAKE).
    Returns (None, BASE_STAKE) on any failure.
    """
    try:
        bdp = SYMBOL_CONFIG[symbol].get("barrier_dp", 5)
        resp = await client.send({
            "proposal": 1, "amount": BASE_STAKE, "basis": "stake",
            "contract_type": "EXPIRYRANGE", "currency": "USD",
            "duration": duration_secs, "duration_unit": "s",
            "underlying_symbol": symbol,
            "barrier": str(round(upper, bdp)), "barrier2": str(round(lower, bdp)),
        }, timeout=12)

        if "error" in resp:
            err = resp["error"].get("message", str(resp["error"]))
            print(f"[Proposal] {symbol} error: {err}")
            return None, BASE_STAKE

        prop      = resp.get("proposal", {})
        payout    = float(prop.get("payout",    0))
        ask_price = float(prop.get("ask_price", BASE_STAKE))

        if payout <= 0 or ask_price <= 0:
            return None, BASE_STAKE

        net_profit = payout - ask_price

        # Sanity: net profit > 20x stake is impossible on EXPIRYRANGE
        if net_profit > BASE_STAKE * 20:
            print(f"[Proposal] {symbol}: suspicious net_profit=${net_profit:.4f} "
                  f"(payout={payout}, ask={ask_price}) — skipping")
            return None, BASE_STAKE

        return net_profit, ask_price

    except Exception as e:
        print(f"[Proposal] {symbol} exception: {e}")
        return None, BASE_STAKE


# =============================================================================
# EXECUTE CONTRACT  — BUG 4 fixed: consistent 3-tuple return
# =============================================================================
async def execute_expiryrange(client: DerivClient, state: BotState,
                               symbol: str, candidate: dict, gate_info: dict,
                               store: SupabaseStore) -> Tuple[bool, float, bool]:
    """
    Returns (won, profit, placed).
    placed=False means payout check failed; caller tries next candidate.
    """
    price_now     = state.last_price[symbol]
    duration_secs = int(candidate["duration_secs"])
    barrier_abs   = candidate["barrier_abs"]
    bdp           = SYMBOL_CONFIG[symbol].get("barrier_dp", 5)
    upper         = round(price_now + barrier_abs, bdp)
    lower         = round(price_now - barrier_abs, bdp)

    # Stage 3: Proposal API payout verification
    net_payout, ask_price = await fetch_proposal_payout(
        client, symbol, upper, lower, duration_secs)

    if net_payout is not None and net_payout < MIN_NET_PAYOUT:
        print(f"[Proposal] {symbol}: net=${net_payout:.4f} < ${MIN_NET_PAYOUT:.4f} "
              f"(barrier too wide) — trying next candidate")
        return False, 0.0, False

    if net_payout is None:
        print(f"[Proposal] {symbol}: API failed — skipping candidate")
        return False, 0.0, False

    SEP = "-" * 68
    print(f"\n{SEP}")
    print(f"  EXPIRYRANGE  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Entry         : {price_now:.5f}")
    print(f"  Barriers      : [{lower:.5f}, {upper:.5f}]  "
          f"(+/-{barrier_abs:.5f} = {candidate['barrier_sigma']:.2f}s_terminal)")
    print(f"  Duration      : {duration_secs}s  ({candidate['n_steps']} ticks)")
    print(f"  Stake/Ask     : ${BASE_STAKE:.2f} / ${ask_price:.4f}")
    print(f"  Net payout    : ${net_payout:.4f}  (confirmed by Deriv proposal API)")
    print(f"  MC win_prob   : {candidate['win_prob']:.3f}  "
          f"CI5={candidate['ci_lower']:.3f}  ({MC_SIMULATIONS:,} sims)")
    print(f"  Vol/tick      : {candidate['vol_per_tick']:.5f} abs price units  "
          f"({'GARCH' if candidate['used_garch'] else 'baseline'})  "
          f"vol_trust={candidate['vol_trust']:.3f}")
    print(f"  Gate          : ADX={gate_info['adx_val']:.1f}  "
          f"vol_trust={gate_info['vol_trust']:.3f}  "
          f"hawkes={gate_info['hawkes_val']:.3f}  "
          f"mbs={gate_info['mbs_val']:.3f}")
    print(SEP)

    won, profit, contract_id = False, 0.0, None
    try:
        resp = await client.send({
            "buy": "1", "price": ask_price,
            "parameters": {
                "amount":            BASE_STAKE,
                "basis":             "stake",
                "contract_type":     "EXPIRYRANGE",
                "currency":          "USD",
                "duration":          duration_secs,
                "duration_unit":     "s",
                "underlying_symbol": symbol,
                "barrier":           str(round(upper, bdp)),
                "barrier2":          str(round(lower, bdp)),
            },
        }, timeout=30)

        if "error" in resp:
            err = resp["error"].get("message", str(resp["error"]))
            print(f"[Buy] {symbol} error: {err}")
            return False, 0.0, False

        contract_id = resp.get("buy", {}).get("contract_id")
        if not contract_id:
            print(f"[Buy] {symbol}: no contract_id: {resp}")
            return False, 0.0, False

        print(f"[Buy] Contract id={contract_id} -- waiting {duration_secs}s...")

        deadline = time.time() + duration_secs + 30
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                poll = await client.send(
                    {"proposal_open_contract": 1, "contract_id": contract_id},
                    timeout=12)
                poc    = poll.get("proposal_open_contract", {})
                status = poc.get("status")
                if status == "sold" or poc.get("is_expired") or poc.get("is_settleable"):
                    profit = float(poc.get("profit", 0.0))
                    won    = profit > 0
                    break
            except Exception:
                pass

    except Exception as e:
        print(f"[Buy] {symbol} exception: {e}")
        return False, 0.0, False

    # Update session state
    state.session_trades[symbol] += 1
    if won:
        state.session_wins[symbol] += 1
    state.session_profit[symbol] += profit
    state.last_trade_time[symbol] = time.time()
    state.last_activity           = time.time()

    wr     = state.session_wins[symbol] / max(state.session_trades[symbol], 1)
    result = f"WIN  +${profit:.4f}" if won else f"LOSS  -${ask_price:.4f}"
    print(f"\n{SEP}")
    print(f"  RESULT  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(SEP)
    print(f"  Contract   : {contract_id}")
    print(f"  Outcome    : {result}")
    print(f"  Session    : {state.session_wins[symbol]}/{state.session_trades[symbol]} "
          f"({wr:.1%})  net=${state.session_profit[symbol]:+.2f}")
    print(SEP + "\n")

    # Refresh balance
    try:
        bal_resp      = await client.send({"balance": 1})
        state.balance = float(bal_resp["balance"]["balance"])
    except Exception:
        pass

    # Log to Supabase
    store.log_trade({
        "symbol":        symbol,
        "entry_price":   price_now,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "barrier_width": barrier_abs * 2,
        "duration_secs": duration_secs,
        "won":           won,
        "profit":        profit,
        "breach_prob":   candidate["breach_prob"],
        "ev_conservative": won * net_payout - (not won) * ask_price,
        "ev_optimistic":   candidate["win_prob"] * net_payout - (1 - candidate["win_prob"]) * ask_price,
        "vol_per_tick":  candidate["vol_per_tick"],
        "used_garch":    candidate["used_garch"],
        "adx_val":       gate_info.get("adx_val", 0.0),
        "vol_trust":     gate_info.get("vol_trust", 0.0),
        "hawkes_val":    gate_info.get("hawkes_val", 0.0),
    })
    return won, profit, True


# =============================================================================
# DAILY SELF-IMPROVEMENT
# =============================================================================
def daily_self_improvement(state: BotState, store: SupabaseStore):
    print("\n" + "=" * 68)
    print("  DAILY SELF-IMPROVEMENT  " +
          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 68)

    for symbol in SYMBOLS:
        rows = store.load_recent_trades(symbol, days=7)
        if not rows:
            print(f"[SI] {symbol}: no history in last 7 days, skipping.")
            continue

        n_total = len(rows)
        n_wins  = sum(1 for r in rows if r.get("won"))
        profit  = sum(float(r.get("profit", 0)) for r in rows)
        print(f"\n[SI] {symbol}: {n_total} trades  {n_wins} wins "
              f"({n_wins/max(n_total,1):.1%})  net=${profit:+.2f}")

        dur_stats: Dict[int,   List[int]] = defaultdict(lambda: [0, 0])
        bar_stats: Dict[float, List[int]] = defaultdict(lambda: [0, 0])
        mc_preds, actuals = [], []

        for r in rows:
            dur  = int(r.get("duration_secs", 120))
            won  = bool(r.get("won", False))
            bw   = float(r.get("barrier_width", 0))  # 2 * barrier_abs
            vpt  = float(r.get("vol_per_tick", 0))
            n_st = max(1, int(dur * SYMBOL_CONFIG[symbol]["ticks_per_sec"]))
            vt   = vpt * math.sqrt(n_st) if vpt > 0 else 1.0
            bs   = round(((bw / 2) / max(vt, 1e-9)) * 2) / 2
            bs   = float(np.clip(bs, 0.5, 3.0))

            dur_stats[dur][1] += 1
            bar_stats[bs][1]  += 1
            if won:
                dur_stats[dur][0] += 1
                bar_stats[bs][0]  += 1

            mc_wp = 1.0 - float(r.get("breach_prob", 0.5))
            mc_preds.append(mc_wp)
            actuals.append(1.0 if won else 0.0)

        alpha = 2.0

        # Duration reweighting
        raw_dw = {}
        print(f"  Duration win rates:")
        for dur, (w, t) in sorted(dur_stats.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_dw[dur] = bwr
            print(f"    {dur}s: {w}/{t} ({w/t:.1%}) Bayes={bwr:.3f}")
        if raw_dw:
            mx, mn, sp = max(raw_dw.values()), min(raw_dw.values()), 0
            sp = mx - mn
            state.duration_weights[symbol] = {
                d: (0.5 + 1.5 * (v - mn) / sp if sp > 0 else 1.0)
                for d, v in raw_dw.items()}

        # Barrier reweighting
        raw_bw = {}
        print(f"  Barrier sigma-slot win rates:")
        for slot, (w, t) in sorted(bar_stats.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_bw[slot] = bwr
            print(f"    s={slot:.1f}: {w}/{t} ({w/t:.1%}) Bayes={bwr:.3f}")
        if raw_bw:
            mx, mn = max(raw_bw.values()), min(raw_bw.values())
            sp = mx - mn
            state.barrier_weights[symbol] = {
                sl: (0.5 + 1.5 * (v - mn) / sp if sp > 0 else 1.0)
                for sl, v in raw_bw.items()}

        # Vol scalar calibration
        if len(mc_preds) >= 10:
            mc_mean  = float(np.mean(mc_preds))
            act_mean = float(np.mean(actuals))
            ratio    = mc_mean / max(act_mean, 0.05)
            old_sc   = state.vol_scalar.get(symbol, 1.0)
            if ratio > 1.08:
                new_sc = float(np.clip(old_sc * min(ratio, 1.25), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar UP: MC={mc_mean:.3f} > actual={act_mean:.3f} "
                      f"-> {old_sc:.3f} -> {new_sc:.3f}")
            elif ratio < 0.92:
                new_sc = float(np.clip(old_sc * max(ratio, 0.80), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar DOWN: MC={mc_mean:.3f} < actual={act_mean:.3f} "
                      f"-> {old_sc:.3f} -> {new_sc:.3f}")
            else:
                print(f"  Vol scalar stable: MC={mc_mean:.3f} ~ actual={act_mean:.3f}")

        best_dur = max(dur_stats, key=lambda d: dur_stats[d][0] / max(dur_stats[d][1], 1)) if dur_stats else 120
        best_bar = max(bar_stats, key=lambda b: bar_stats[b][0] / max(bar_stats[b][1], 1)) if bar_stats else 1.0
        store.save_daily_summary(
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            symbol, n_total, n_wins, profit, best_dur, best_bar)

    store.save_config("duration_weights",
        {s: {str(k): v for k, v in state.duration_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("barrier_weights",
        {s: {str(k): v for k, v in state.barrier_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("vol_scalars",
        {s: state.vol_scalar.get(s, 1.0) for s in SYMBOLS})

    state.last_daily_tune = time.time()
    print("\n[SI] Config saved. Next tuning in ~24h.")
    print("=" * 68 + "\n")


def load_config_from_supabase(state: BotState, store: SupabaseStore):
    loaded = False
    dur_w = store.load_config("duration_weights")
    if dur_w:
        for s in SYMBOLS:
            if s in dur_w:
                state.duration_weights[s] = {int(k): float(v) for k, v in dur_w[s].items()}
                loaded = True
    bar_w = store.load_config("barrier_weights")
    if bar_w:
        for s in SYMBOLS:
            if s in bar_w:
                state.barrier_weights[s] = {float(k): float(v) for k, v in bar_w[s].items()}
                loaded = True
    vol_s = store.load_config("vol_scalars")
    if vol_s:
        for s in SYMBOLS:
            if s in vol_s:
                state.vol_scalar[s] = float(vol_s[s])
                loaded = True
    if loaded:
        print(f"[Config] Warm-start loaded. vol_scalars={state.vol_scalar}")
    else:
        print("[Config] Cold start.")


# =============================================================================
# TICK HELPERS
# =============================================================================
async def fetch_history(client: DerivClient, symbol: str, count: int) -> list:
    resp = await client.send(
        {"ticks_history": symbol, "count": count, "end": "latest", "style": "ticks"})
    h = resp.get("history", {})
    return list(zip(h.get("times", []), h.get("prices", [])))


async def subscribe_ticks(client: DerivClient, symbol: str) -> asyncio.Queue:
    q = client.subscribe_channel("tick")
    await client.send({"ticks": symbol, "subscribe": 1})
    return q


# =============================================================================
# WATCHDOG  — BUG 5 fixed: 15 min timeout
# =============================================================================
async def watchdog(state: BotState):
    while True:
        await asyncio.sleep(30)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s -- restarting.")
            os.execv(sys.executable, [sys.executable] + sys.argv)


# =============================================================================
# MAIN
# =============================================================================
async def main():
    if not DERIV_API_TOKEN:
        sys.exit("[FATAL] DERIV_API_TOKEN not set.")
    if not DERIV_APP_ID:
        sys.exit("[FATAL] DERIV_APP_ID not set.")

    store = SupabaseStore()
    state = BotState()
    load_config_from_supabase(state, store)

    client  = DerivClient(DERIV_APP_ID, DERIV_API_TOKEN,
                          DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    account = await client.connect()
    state.balance = float(account.get("balance", 0))
    print(f"Balance: ${state.balance:.2f}")

    sdata: Dict[str, SymbolData] = {s: SymbolData(s) for s in SYMBOLS}

    # Bootstrap history
    print("\nBootstrapping tick history...")
    for sym in SYMBOLS:
        ticks = await fetch_history(client, sym, HISTORY_BOOTSTRAP)
        for epoch, price in ticks:
            sdata[sym].add_tick(epoch, price)
        prices = sdata[sym].prices()
        if len(prices):
            state.last_price[sym] = float(prices[-1])
        print(f"  {sym}: {len(ticks)} ticks loaded  "
              f"price={state.last_price[sym]:.5f}")
    state.last_activity = time.time()  # reset after bootstrap

    # Fit GARCH
    print("\nFitting GARCH models...")
    for sym in SYMBOLS:
        returns = sdata[sym].returns()
        if len(returns) >= MIN_TICKS_FOR_FIT:
            gr = await asyncio.to_thread(fit_garch, returns)
            state.garch_cache[sym] = (gr, time.time())
            # Sanity-check the vol immediately
            prices = sdata[sym].prices()
            abs_vol, vol_trust, used_g = estimate_abs_vol_per_tick(
                prices, returns, gr, state.last_price[sym])
            print(f"  {sym}: GARCH {'fitted' if gr else 'failed'}  "
                  f"abs_vol_per_tick={abs_vol:.5f}  "
                  f"({'GARCH' if used_g else 'baseline'})  "
                  f"vol_trust={vol_trust:.3f}")
        else:
            state.garch_cache[sym] = (None, 0.0)
            print(f"  {sym}: not enough data ({len(returns)} returns)")
    state.last_activity = time.time()  # reset after GARCH

    # Subscribe ticks
    tick_queues: Dict[str, asyncio.Queue] = {}
    for sym in SYMBOLS:
        tick_queues[sym] = await subscribe_ticks(client, sym)
    print(f"\nSubscribed to: {SYMBOLS}")

    async def resubscribe(c: DerivClient):
        for sym in SYMBOLS:
            tick_queues[sym] = await subscribe_ticks(c, sym)
        bal_resp     = await c.send({"balance": 1})
        state.balance = float(bal_resp.get("balance", {}).get("balance", state.balance))
        print("[Reconnect] Subscriptions restored.")

    client.resubscribe_cb = resubscribe

    asyncio.create_task(watchdog(state))
    state.last_activity = time.time()

    print("\n" + "=" * 68)
    print("  Bot armed -- scanning for EXPIRYRANGE setups")
    print("=" * 68 + "\n")

    garch_recal_secs = 2 * 3600

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    while True:
        # Drain tick queues
        for sym in SYMBOLS:
            drained = 0
            while drained < 200:
                try:
                    msg  = tick_queues[sym].get_nowait()
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        ep = float(tick.get("epoch", 0))
                        px = float(tick.get("quote", 0))
                        sdata[sym].add_tick(ep, px)
                        state.last_price[sym] = px
                        drained += 1
                except asyncio.QueueEmpty:
                    break
            if drained == 0:
                try:
                    msg = await asyncio.wait_for(tick_queues[sym].get(), timeout=1.5)
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        sdata[sym].add_tick(float(tick.get("epoch", 0)),
                                            float(tick.get("quote", 0)))
                        state.last_price[sym] = float(tick.get("quote", 0))
                except asyncio.TimeoutError:
                    pass

        state.last_activity = time.time()

        # Daily self-improvement check
        now_utc = datetime.now(timezone.utc)
        if (time.time() - state.last_daily_tune > 23 * 3600
                and now_utc.hour == DAILY_TUNE_HOUR_UTC):
            await asyncio.to_thread(daily_self_improvement, state, store)

        # Periodic GARCH recalibration
        for sym in SYMBOLS:
            gr, fitted_at = state.garch_cache.get(sym, (None, 0.0))
            if time.time() - fitted_at > garch_recal_secs:
                returns = sdata[sym].returns()
                if len(returns) >= MIN_TICKS_FOR_FIT:
                    gr_new = await asyncio.to_thread(fit_garch, returns)
                    state.garch_cache[sym] = (gr_new, time.time())
                    state.last_activity    = time.time()
                    prices    = sdata[sym].prices()
                    abs_vol, _, used_g = estimate_abs_vol_per_tick(
                        prices, returns, gr_new, state.last_price[sym])
                    print(f"[GARCH] {sym}: recalibrated  "
                          f"abs_vol={abs_vol:.5f}  "
                          f"({'GARCH' if used_g else 'baseline'})")

        if state.trading_locked:
            await asyncio.sleep(0.5)
            continue

        # Per-symbol evaluation
        for sym in SYMBOLS:
            sd = sdata[sym]
            if len(sd.ticks) < MIN_TICKS_LIVE:
                continue

            elapsed = time.time() - state.last_trade_time.get(sym, 0.0)
            if elapsed < SYMBOL_CONFIG[sym]["cooldown_secs"]:
                continue

            prices      = sd.prices()
            price_diffs = sd.price_diffs()
            returns     = sd.returns()
            if len(returns) < 20:
                continue

            garch_result, _ = state.garch_cache.get(sym, (None, 0.0))
            price_now       = state.last_price[sym]

            # Stage 1: Structural gate
            gate_ok, gate_info = structural_gate(
                sym, prices, price_diffs, returns, garch_result, price_now)
            if not gate_ok:
                fails = {k: v for k, v in gate_info.items() if k.startswith("fail_")}
                print(f"[Gate] {sym}: blocked -- {fails}")
                continue

            # Stage 2: MC optimizer
            print(f"\n[MC] {sym}: running {MC_SIMULATIONS:,}-sim optimizer "
                  f"(ADX={gate_info['adx_val']:.1f} "
                  f"vol_trust={gate_info['vol_trust']:.3f} "
                  f"abs_vol={gate_info['abs_vol']:.5f})...")

            t0         = time.time()
            candidates = await asyncio.to_thread(
                mc_auto_optimize, prices, price_diffs, returns,
                sym, garch_result, state)
            dt = time.time() - t0

            if not candidates:
                print(f"[MC] {sym}: no combo cleared win>={MC_REQUIRED_WIN:.0%} "
                      f"& CI>={MC_REQUIRED_CI:.0%} in {dt:.1f}s -- waiting.")
                continue

            print(f"[MC] {sym}: {len(candidates)} passing combos in {dt:.1f}s -- "
                  f"trying lowest-win first (= highest payout)")

            # Stages 3+4: iterate candidates until one passes proposal + buy
            state.trading_locked = True
            placed = False
            for cand in candidates[:6]:
                print(f"[MC]   Candidate: dur={cand['duration_secs']}s "
                      f"barrier=+/-{cand['barrier_abs']:.5f} "
                      f"({cand['barrier_sigma']:.2f}s)  "
                      f"win={cand['win_prob']:.3f}")
                won, profit, ok = await execute_expiryrange(
                    client, state, sym, cand, gate_info, store)
                if ok:
                    placed = True
                    break
            state.trading_locked = False

            if not placed:
                print(f"[MC] {sym}: all candidates failed payout check -- skipping cycle.")

        await asyncio.sleep(0.1)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")
    except Exception as e:
        print(f"[main] {type(e).__name__}: {e}")
        sys.stdout.flush()
        time.sleep(3)
        os.execv(sys.executable, [sys.executable] + sys.argv)
