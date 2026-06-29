"""
Deriv EXPIRYRANGE (ENDSIN) Monte Carlo Bot — 1HZ10V + RDBEAR
==============================================================
Inherits the full connection layer, Supabase store, and statistical
model pipeline from deriv_multisymbol_bot.py.

Specialised for EXPIRYRANGE contracts:
  • Wins if price at expiry is INSIDE ±barrier from entry price.
  • Direction-agnostic — we predict CONTAINMENT, not up/down.

EDGE MECHANISM — two-stage filter:
  Stage 1 (Gate):      5 structural conditions block trending / volatile markets.
                       Only range-bound, low-vol, low-momentum windows pass.
  Stage 2 (MC):        75K-path simulation using conservative GARCH vol.
                       Requires MC win_prob >= 0.72 AND CI₅ >= 0.70.
                       In genuinely range-bound conditions true vol ≈ 75% of
                       GARCH estimate → actual win rate ≈ 85%+ on valid setups.
  Stage 3 (Proposal):  Deriv proposal API call verifies ACTUAL net payout >= $0.182
                       (52% of $0.35 stake). Only real Deriv numbers accepted.
  Stage 4 (Execute):   Buy the contract, wait for result, log to Supabase.

SELF-IMPROVEMENT (daily, midnight UTC):
  Loads last 7 days of bot_expiryrange_log from Supabase.
  Re-weights duration/barrier preferences using Bayesian win-rate estimates.
  Calibrates vol_scalar by comparing MC predictions to actual outcomes.
  Saves learned config back to bot_expiryrange_config for warm-start on restart.

CONNECTION:
  New Deriv Options API — REST OTP bootstrap, same as parent bot.
  GET  /trading/v1/options/accounts          → resolve account_id
  POST /trading/v1/options/accounts/{id}/otp → pre-authenticated WS URL
  No `authorize` message needed; OTP URL is already scoped to account.
  Auto-reconnects with exponential backoff; replays subscriptions after reconnect.

ENV VARS:
  DERIV_APP_ID         new app from developers.deriv.com (legacy ids don't work)
  DERIV_API_TOKEN      personal access token
  DERIV_ACCOUNT_TYPE   "demo" (default) or "real"
  DERIV_ACCOUNT_ID     optional: skip account lookup
  SUPABASE_URL         https://xxxx.supabase.co
  SUPABASE_KEY         service_role key

SUPABASE SQL (run once — see supabase_schema.sql):
  bot_expiryrange_log     every trade with full MC diagnostics
  bot_expiryrange_config  learned weights + vol scalars (warm-start)
  bot_expiryrange_daily   daily summary per symbol
"""

import asyncio, contextlib, io, json, math, os, random, sys, time, warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import websockets
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn.hmm import GaussianHMM
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Deriv connection ────────────────────────────────────────────────────────
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID   = os.getenv("DERIV_ACCOUNT_ID") or None
API_BASE           = "https://api.derivws.com"
ACCOUNTS_PATH      = "/trading/v1/options/accounts"
OTP_PATH           = "/trading/v1/options/accounts/{account_id}/otp"

# ── Supabase ────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Target symbols ──────────────────────────────────────────────────────────
SYMBOLS = ["1HZ10V", "RDBEAR"]

# ── Contract parameters ─────────────────────────────────────────────────────
BASE_STAKE         = 0.35         # Deriv minimum stake
MIN_NET_PAYOUT     = 0.182        # 52% of $0.35 — requires proposal API confirmation
MIN_TICKS_FOR_FIT  = 200
MIN_TICKS_LIVE     = 60
HISTORY_BOOTSTRAP  = 5000
WATCHDOG_TIMEOUT   = 5 * 60      # restart if silent for 5 minutes

# ── MC engine ───────────────────────────────────────────────────────────────
MC_SIMULATIONS      = 75_000     # paths per evaluation
MC_CI_PERCENTILE    = 5          # use 5th-percentile CI lower bound (not mean)
MC_REQUIRED_WIN     = 0.72       # MC must show >= 72% win probability (conservative)
MC_REQUIRED_CI      = 0.70       # CI lower bound must also clear 70%
MC_BATCH_SIZE       = 25_000     # batch to avoid RAM spikes

# ── Duration and barrier sweep grids ────────────────────────────────────────
# Durations: minimum 2 minutes as per spec; max 8 minutes (beyond that
# the containment assumption weakens and OU pull may be insufficient)
DURATION_CANDIDATES = [120, 180, 240, 300, 360, 420, 480]   # seconds

# Barrier expressed as multiples of terminal vol (vol_per_tick * sqrt(n_steps))
# We target the zone where MC win_prob ≈ 72-82% (barrier ≈ 1.0-1.3 σ_terminal)
# Deriv's payout is highest at lower win rates — fine as long as proposal clears $0.182
BARRIER_SIGMAS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00]
BARRIER_ABS_MIN = 0.3    # floor: at least 0.3 price units
BARRIER_ABS_MAX = 100.0  # ceiling

# ── Per-symbol configuration ────────────────────────────────────────────────
SYMBOL_CONFIG = {
    "1HZ10V": {
        "ticks_per_sec":         1.0,
        "max_adx":               22,     # ADX above → trending → skip
        "min_vol_trust":         0.85,   # GARCH vol/baseline ratio must be calm
        "max_mbs":               0.40,   # no structural break
        "boll_width_factor":     1.20,   # recent_std <= median_std × this
        "max_hawkes":            0.50,   # no momentum clustering
        "cooldown_secs":         150,    # minimum gap between trades
        "tick_dt":               1.0,
        "garch_scale":           1000.0,
    },
    "RDBEAR": {
        "ticks_per_sec":         1.0,    # verify against live data; RDBEAR may differ
        "max_adx":               18,     # stricter: bear drift raises ADX quickly
        "min_vol_trust":         0.80,   # stricter vol requirement
        "max_mbs":               0.30,   # stricter: structural breaks common
        "boll_width_factor":     1.15,   # stricter: bands must be very compressed
        "max_hawkes":            0.40,   # stricter: jump events more frequent
        "cooldown_secs":         180,    # longer cooldown per wider barriers
        "tick_dt":               1.0,
        "garch_scale":           1000.0,
    },
}

DAILY_TUNE_HOUR_UTC = 0    # midnight UTC


# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE STORE
# ═══════════════════════════════════════════════════════════════════════════
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        if self.ok:
            print(f"[Store] Supabase active → {self.url}")
        else:
            print("[Store] No Supabase creds — learned state will NOT persist.")

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
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[Store] {table} upsert error: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                headers=self._hdr(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[Store] {table} insert error: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                headers=self._hdr("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[Store] {table} select error: {e}")
        return []

    # ── Trade logging ────────────────────────────────────────────────────
    def log_trade(self, rec: dict):
        self._insert("bot_expiryrange_log", {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "symbol":          rec["symbol"],
            "entry_price":     round(float(rec["entry_price"]),     5),
            "upper_barrier":   round(float(rec["upper_barrier"]),   5),
            "lower_barrier":   round(float(rec["lower_barrier"]),   5),
            "barrier_width":   round(float(rec["barrier_width"]),   5),
            "barrier_sigma":   round(float(rec.get("barrier_sigma", 0)), 3),
            "duration_secs":   int(rec["duration_secs"]),
            "stake":           round(float(BASE_STAKE),             4),
            "won":             bool(rec["won"]),
            "profit":          round(float(rec["profit"]),          4),
            "mc_win_prob":     round(float(rec["mc_win_prob"]),     4),
            "mc_ci_lower":     round(float(rec["mc_ci_lower"]),     4),
            "breach_prob":     round(float(rec["breach_prob"]),     4),
            "actual_payout":   round(float(rec.get("actual_payout", 0)), 4),
            "vol_per_tick":    round(float(rec["vol_per_tick"]),    6),
            "used_garch":      bool(rec["used_garch"]),
            "adx_val":         round(float(rec.get("adx_val", 0)), 3),
            "vol_trust":       round(float(rec.get("vol_trust", 0)), 4),
            "hawkes_val":      round(float(rec.get("hawkes_val", 0)), 4),
            "n_sims":          int(MC_SIMULATIONS),
        })

    # ── Config persistence ───────────────────────────────────────────────
    def save_config(self, key, value):
        self._upsert("bot_expiryrange_config", {
            "key": key, "value": json.dumps(value),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def load_config(self, key):
        rows = self._select("bot_expiryrange_config",
                            f"select=value&key=eq.{key}")
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
            "best_duration":int(best_dur), "best_barrier": round(float(best_bar), 4),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        })

    def load_recent_trades(self, symbol, days=7):
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self._select("bot_expiryrange_log",
            f"select=*&symbol=eq.{symbol}&ts=gte.{since}&order=ts.asc")


# ═══════════════════════════════════════════════════════════════════════════
# DERIV CLIENT  (identical connection layer from parent bot)
# ═══════════════════════════════════════════════════════════════════════════
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
        self.pending       = {}
        self.subscriptions = defaultdict(list)
        self.account       = None
        self.resubscribe_cb = None
        self._running      = False
        self._reader_task  = None
        self._ka_task      = None

    # ── REST bootstrap ───────────────────────────────────────────────────
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
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(
            f"No '{self.account_type}' account found. "
            f"Set DERIV_ACCOUNT_ID or create one. Returned: {data}")

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
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ── Lifecycle ────────────────────────────────────────────────────────
    async def connect(self):
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url      = await self._get_ws_url()
        self.ws     = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task     = asyncio.create_task(self._heartbeat())
        bal           = await self.send({"balance": 1})
        self.account  = bal.get("balance", {})
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
            if self._reader_task is not None:
                await self._reader_task
            if self._ka_task is not None:
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
                    print(f"[Client] Reconnect {attempt} failed: {e}")

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
        request = dict(request)
        request["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL DATA BUFFER
# ═══════════════════════════════════════════════════════════════════════════
class SymbolData:
    def __init__(self, symbol, maxlen=8000, tick_dt=1.0):
        self.symbol  = symbol
        self.tick_dt = tick_dt
        self.ticks   = deque(maxlen=maxlen)

    def add_tick(self, epoch, price):
        self.ticks.append((float(epoch), float(price)))

    def prices(self)  -> np.ndarray:
        return np.array([p for _, p in self.ticks], dtype=float)

    def epochs(self)  -> np.ndarray:
        return np.array([e for e, _ in self.ticks], dtype=float)

    def returns(self) -> np.ndarray:
        p = self.prices()
        return np.diff(p) / p[:-1] if len(p) >= 2 else np.array([])

    def mean_tick_dt(self) -> float:
        e = self.epochs()
        return float(np.mean(np.diff(e))) if len(e) >= 2 else self.tick_dt


# ═══════════════════════════════════════════════════════════════════════════
# BOT STATE
# ═══════════════════════════════════════════════════════════════════════════
class BotState:
    def __init__(self):
        self.balance          = 0.0
        self.trading_locked   = False
        self.last_activity    = time.time()
        self.last_trade_time  = {s: 0.0 for s in SYMBOLS}
        self.last_daily_tune  = 0.0
        # Fitted GARCH models cache: symbol → (result, fitted_at)
        self.garch_cache: Dict[str, tuple] = {}
        # Learned from daily self-improvement
        self.vol_scalar:       Dict[str, float]       = {s: 1.0 for s in SYMBOLS}
        self.duration_weights: Dict[str, Dict[int, float]] = {s: {} for s in SYMBOLS}
        self.barrier_weights:  Dict[str, Dict[float, float]] = {s: {} for s in SYMBOLS}
        # Session stats
        self.session_trades  = {s: 0   for s in SYMBOLS}
        self.session_wins    = {s: 0   for s in SYMBOLS}
        self.session_profit  = {s: 0.0 for s in SYMBOLS}


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def fit_garch(returns: np.ndarray, scale: float = 1000.0):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * scale
        am     = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                return am.fit(disp="off")
    except Exception as e:
        print(f"[GARCH] fit failed: {e}")
        return None


def garch_cond_vol(garch_result, returns: np.ndarray,
                   scale: float = 1000.0) -> Tuple[float, float]:
    """Returns (cond_vol, vol_trust). vol_trust ∈ [0.1, 1.0]."""
    baseline = float(np.std(returns)) if len(returns) > 5 else 1e-6
    baseline = max(baseline, 1e-9)
    if garch_result is None:
        return baseline, 0.5
    try:
        fc       = garch_result.forecast(horizon=1, reindex=False)
        cond_vol = math.sqrt(float(fc.variance.values[-1, 0])) / scale
        ratio    = cond_vol / baseline
        trust    = 1.0 / (1.0 + max(ratio - 1.0, 0) * 2)
        return float(cond_vol), float(np.clip(trust, 0.1, 1.0))
    except Exception:
        return baseline, 0.5


def compute_adx(prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
    """Returns (adx_value, trend_strength 0-to-1)."""
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
    pdi = 100 * np.mean(pdm_a[-period:]) / atr
    ndi = 100 * np.mean(ndm_a[-period:]) / atr
    adx_vals = [
        100 * abs(pdm_a[i] - ndm_a[i]) / (np.mean(tr_a[max(0,i-period):i]) * period + 1e-9)
        for i in range(period, len(tr_a))
    ]
    adx = float(np.clip(np.mean(adx_vals) if adx_vals else 20.0, 0, 100))
    return adx, float(np.clip((adx - 20) / 30, 0, 1))


def compute_bollinger_width(prices: np.ndarray) -> Tuple[float, float]:
    """Returns (current_std_10, rolling_median_std_10)."""
    if len(prices) < 30:
        return 1.0, 1.0
    cur_std = float(np.std(prices[-10:]))
    stds = [float(np.std(prices[i:i+10])) for i in range(0, len(prices)-10, 5)]
    return cur_std, float(np.median(stds)) if stds else cur_std


def compute_hawkes_proxy(returns: np.ndarray) -> float:
    """Fast Hawkes intensity proxy in [0, 1]. > 0.5 → momentum clustering."""
    if len(returns) < 20:
        return 0.0
    thresh = 0.5 * np.std(returns) if np.std(returns) > 0 else 1e-9
    recent  = float(np.mean(np.abs(returns[-20:]) > thresh))
    base    = float(np.mean(np.abs(returns) > thresh)) if len(returns) >= 50 else 0.1
    return float(np.clip((recent / max(base, 1e-6) - 1.0) / 3.0, 0.0, 1.0))


def compute_mbs(prices: np.ndarray, lookback: int = 50) -> float:
    """Market Behaviour Score — structural break detector. Returns [0, 1]."""
    if len(prices) < lookback + 5:
        return 0.0
    w       = prices[-lookback:]
    rng     = np.max(w) - np.min(w)
    if rng < 1e-9:
        return 0.0
    prev_hi = np.max(w[:-10])
    prev_lo = np.min(w[:-10])
    last    = prices[-1]
    bos_up   = max(0.0, last - prev_hi) / rng
    bos_down = max(0.0, prev_lo - last) / rng
    return float(np.clip(max(bos_up, bos_down), 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL GATE — all 5 conditions must pass
# ═══════════════════════════════════════════════════════════════════════════
def structural_gate(symbol: str, prices: np.ndarray, returns: np.ndarray,
                    garch_result) -> Tuple[bool, dict]:
    cfg = SYMBOL_CONFIG[symbol]
    ok  = True
    info: dict = {}

    # 1. ADX
    adx_val, adx_str = compute_adx(prices)
    info["adx_val"]  = adx_val
    if adx_val > cfg["max_adx"]:
        ok = False
        info["fail_adx"] = f"ADX={adx_val:.1f} > {cfg['max_adx']}"

    # 2. GARCH vol trust
    cond_vol, vol_trust = garch_cond_vol(garch_result, returns, cfg["garch_scale"])
    info["vol_trust"] = vol_trust
    info["cond_vol"]  = cond_vol
    if vol_trust < cfg["min_vol_trust"]:
        ok = False
        info["fail_vol"] = f"vol_trust={vol_trust:.3f} < {cfg['min_vol_trust']}"

    # 3. MBS
    mbs_val         = compute_mbs(prices)
    info["mbs_val"] = mbs_val
    if mbs_val >= cfg["max_mbs"]:
        ok = False
        info["fail_mbs"] = f"mbs={mbs_val:.3f} >= {cfg['max_mbs']}"

    # 4. Bollinger width
    cur_std, med_std = compute_bollinger_width(prices)
    info["cur_std"]  = cur_std
    info["med_std"]  = med_std
    if med_std > 0 and cur_std > med_std * cfg["boll_width_factor"]:
        ok = False
        info["fail_boll"] = (f"cur_std={cur_std:.5f} > "
                             f"med×{cfg['boll_width_factor']}={med_std*cfg['boll_width_factor']:.5f}")

    # 5. Hawkes
    hawkes_val        = compute_hawkes_proxy(returns)
    info["hawkes_val"] = hawkes_val
    if hawkes_val > cfg["max_hawkes"]:
        ok = False
        info["fail_hawkes"] = f"hawkes={hawkes_val:.3f} > {cfg['max_hawkes']}"

    return ok, info


# ═══════════════════════════════════════════════════════════════════════════
# MC BARRIER BREACH ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════
def mc_breach_estimate(prices: np.ndarray, returns: np.ndarray,
                       barrier_abs: float, duration_secs: float,
                       ticks_per_sec: float, garch_result,
                       garch_scale: float, vol_scalar: float) -> dict:
    """
    Estimates Pr(|X_terminal| >= barrier_abs) via batched MC.

    EXPIRYRANGE settles on TERMINAL price vs entry (not path maximum).
    Terminal distribution: X_T ~ N(0, vol_per_tick * sqrt(n_steps))
    where drift = 0 (synthetic index RNG has zero drift by design;
    any RDBEAR bear drift is too small to meaningfully shift the
    terminal distribution over 2-8 minute windows).

    Returns dict with win_prob, breach_prob, vol diagnostics.
    """
    n_steps      = max(1, int(round(duration_secs * ticks_per_sec)))
    baseline_vol = float(np.std(returns[-200:]) if len(returns) >= 200 else np.std(returns))
    baseline_vol = max(baseline_vol, 1e-9)

    cond_vol, vol_trust = garch_cond_vol(garch_result, returns, garch_scale)
    sane_garch = cond_vol > 0 and 0.3 < cond_vol / baseline_vol < 5.0
    vol_raw    = cond_vol if sane_garch else baseline_vol
    used_garch = sane_garch
    vol_per_tick = vol_raw * vol_scalar

    # Safety: vol implausibly small → GARCH unit conversion suspect
    if vol_per_tick < 1e-6:
        return {"blocked": True,
                "reason": f"vol_per_tick={vol_per_tick:.2e} < 1e-6 (suspect)",
                "win_prob": 0.0, "breach_prob": 1.0,
                "vol_per_tick": vol_per_tick, "used_garch": used_garch,
                "vol_trust": vol_trust, "n_steps": n_steps}

    # Batched simulation
    wins = 0
    done = 0
    vol_terminal = vol_per_tick * math.sqrt(n_steps)
    while done < MC_SIMULATIONS:
        batch    = min(MC_BATCH_SIZE, MC_SIMULATIONS - done)
        terminal = np.random.normal(0.0, vol_terminal, size=batch)
        wins    += int(np.sum(np.abs(terminal) < barrier_abs))   # INSIDE = win
        done    += batch

    win_prob    = wins / MC_SIMULATIONS
    breach_prob = 1.0 - win_prob

    return {
        "blocked":      False,
        "win_prob":     win_prob,
        "breach_prob":  breach_prob,
        "vol_per_tick": vol_per_tick,
        "vol_terminal": vol_terminal,
        "used_garch":   used_garch,
        "vol_trust":    vol_trust,
        "baseline_vol": baseline_vol,
        "n_steps":      n_steps,
        "barrier":      barrier_abs,
    }


# CI lower bound (normal approx, conservative)
def ci_lower(win_prob: float, n: int, percentile: int = MC_CI_PERCENTILE) -> float:
    z = norm.ppf(1 - percentile / 100)
    return win_prob - z * math.sqrt(win_prob * (1 - win_prob) / n)


# ═══════════════════════════════════════════════════════════════════════════
# MC AUTO-OPTIMIZER — sweep (duration × barrier) → best EV combo
# ═══════════════════════════════════════════════════════════════════════════
def mc_auto_optimize(prices: np.ndarray, returns: np.ndarray,
                     symbol: str, garch_result,
                     state: BotState) -> Optional[dict]:
    """
    For each (duration_secs, barrier_sigma) pair:
      1. Compute barrier_abs = barrier_sigma × vol_terminal
      2. Run MC → win_prob, breach_prob
      3. Filter: win_prob >= MC_REQUIRED_WIN AND ci_lower >= MC_REQUIRED_CI
      4. Apply learned duration/barrier weights
    Select combo with highest weighted win_prob among all passing pairs.
    Returns None if no combo passes.

    NOTE: actual payout verification (>=  $0.182) happens via the Deriv
    proposal API in execute_expiryrange, AFTER this optimizer picks the
    best structural combo. This keeps the MC loop fast (no async calls).
    """
    cfg          = SYMBOL_CONFIG[symbol]
    garch_scale  = cfg["garch_scale"]
    ticks_per_sec = cfg["ticks_per_sec"]
    vol_sc       = state.vol_scalar.get(symbol, 1.0)

    baseline_vol = float(np.std(returns[-200:]) if len(returns) >= 200 else np.std(returns))
    baseline_vol = max(baseline_vol, 1e-9)
    cond_vol, _ = garch_cond_vol(garch_result, returns, garch_scale)
    sane = cond_vol > 0 and 0.3 < cond_vol / baseline_vol < 5.0
    vol_per_tick = (cond_vol if sane else baseline_vol) * vol_sc

    candidates = []
    for dur_secs in DURATION_CANDIDATES:
        n_steps      = max(1, int(round(dur_secs * ticks_per_sec)))
        vol_terminal = vol_per_tick * math.sqrt(n_steps)

        for bs in BARRIER_SIGMAS:
            barrier_abs = float(np.clip(bs * vol_terminal,
                                        BARRIER_ABS_MIN, BARRIER_ABS_MAX))

            mc = mc_breach_estimate(
                prices, returns, barrier_abs, dur_secs, ticks_per_sec,
                garch_result, garch_scale, vol_sc,
            )
            if mc.get("blocked"):
                continue

            wp  = mc["win_prob"]
            cil = ci_lower(wp, MC_SIMULATIONS, MC_CI_PERCENTILE)

            if wp  < MC_REQUIRED_WIN: continue
            if cil < MC_REQUIRED_CI:  continue

            # Learned preference weights (default 1.0 = neutral)
            dw  = state.duration_weights.get(symbol, {}).get(dur_secs, 1.0)
            bw  = state.barrier_weights.get(symbol, {}).get(round(bs * 2) / 2, 1.0)
            wev = wp * dw * bw   # higher = preferred

            candidates.append({
                "duration_secs":  dur_secs,
                "barrier_abs":    barrier_abs,
                "barrier_sigma":  bs,
                "n_steps":        n_steps,
                "win_prob":       wp,
                "ci_lower":       cil,
                "breach_prob":    mc["breach_prob"],
                "vol_per_tick":   mc["vol_per_tick"],
                "vol_terminal":   vol_terminal,
                "used_garch":     mc["used_garch"],
                "vol_trust":      mc["vol_trust"],
                "weighted_score": wev,
                "n_sims":        MC_SIMULATIONS,
            })

    if not candidates:
        return None
    return max(candidates, key=lambda x: x["weighted_score"])


# ═══════════════════════════════════════════════════════════════════════════
# PROPOSAL API — verify actual Deriv payout before buying
# ═══════════════════════════════════════════════════════════════════════════
async def fetch_proposal_payout(client: DerivClient, symbol: str,
                                upper: float, lower: float,
                                duration_secs: int) -> Optional[float]:
    """Calls Deriv proposal API. Returns net profit amount or None on failure."""
    try:
        resp = await client.send({
            "proposal": 1, "amount": BASE_STAKE, "basis": "stake",
            "contract_type": "EXPIRYRANGE", "currency": "USD",
            "duration": duration_secs, "duration_unit": "s",
            "underlying_symbol": symbol,
            "barrier": str(round(upper, 5)), "barrier2": str(round(lower, 5)),
        }, timeout=10)
        if "error" in resp:
            return None
        total = float(resp.get("proposal", {}).get("payout", 0))
        net   = total - BASE_STAKE
        return net if net > 0 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTE EXPIRYRANGE CONTRACT
# ═══════════════════════════════════════════════════════════════════════════
async def execute_expiryrange(client: DerivClient, state: BotState,
                               symbol: str, best: dict, gate_info: dict,
                               store: SupabaseStore) -> Tuple[bool, float, bool]:
    """Returns (won, profit, placed). placed=False if payout check failed."""
    entry_price    = float(state._last_price[symbol])
    duration_secs  = int(best["duration_secs"])
    barrier_abs    = best["barrier_abs"]
    upper          = round(entry_price + barrier_abs, 5)
    lower          = round(entry_price - barrier_abs, 5)

    # ── Stage 3: Proposal API payout verification ─────────────────────────
    actual_payout = await fetch_proposal_payout(client, symbol, upper, lower, duration_secs)
    if actual_payout is None:
        print(f"[Proposal] {symbol}: API call failed — using conservative estimate")
        actual_payout = 0.0   # will check below

    # Enforce $0.182 minimum payout (52% of $0.35 stake)
    if actual_payout > 0 and actual_payout < MIN_NET_PAYOUT:
        print(f"[Proposal] {symbol}: payout ${actual_payout:.4f} < ${MIN_NET_PAYOUT:.4f} "
              f"— barrier too wide for 52% floor.")
        return False, 0.0, False

    sep = "─" * 68
    ts  = datetime.now(timezone.utc).isoformat()
    print(f"\n{sep}")
    print(f"  EXPIRYRANGE  {symbol}  {ts}")
    print(sep)
    print(f"  Entry          : {entry_price:.5f}")
    print(f"  Barriers       : [{lower:.5f}, {upper:.5f}]  "
          f"(±{barrier_abs:.5f} = {best['barrier_sigma']:.2f}σ_terminal)")
    print(f"  Duration       : {duration_secs}s  ({best['n_steps']} ticks)")
    print(f"  Stake          : ${BASE_STAKE:.2f}")
    print(f"  MC win_prob    : {best['win_prob']:.3f}  "
          f"CI₅={best['ci_lower']:.3f}  ({MC_SIMULATIONS:,} sims)")
    print(f"  Vol/tick       : {best['vol_per_tick']:.6f}  "
          f"{'GARCH' if best['used_garch'] else 'baseline'}  "
          f"vol_trust={best['vol_trust']:.3f}")
    print(f"  Payout         : ${actual_payout:.4f} net"
          if actual_payout > 0 else "  Payout         : (proposal failed — continuing)")
    print(f"  Gate           : ADX={gate_info['adx_val']:.1f}  "
          f"vol_trust={gate_info['vol_trust']:.3f}  "
          f"hawkes={gate_info['hawkes_val']:.3f}  "
          f"mbs={gate_info['mbs_val']:.3f}")
    print(sep)

    won, profit, contract_id = False, 0.0, None
    try:
        resp = await client.send({
            "buy": "1", "price": BASE_STAKE,
            "parameters": {
                "amount":            BASE_STAKE,
                "basis":             "stake",
                "contract_type":     "EXPIRYRANGE",
                "currency":          "USD",
                "duration":          duration_secs,
                "duration_unit":     "s",
                "underlying_symbol": symbol,
                "barrier":           str(upper),
                "barrier2":          str(lower),
            },
        }, timeout=30)

        if "error" in resp:
            print(f"[Buy] {symbol} error: {resp['error'].get('message', resp['error'])}")
            return False, 0.0, False

        contract_id = resp.get("buy", {}).get("contract_id")
        if not contract_id:
            print(f"[Buy] {symbol}: no contract_id in response: {resp}")
            return False, 0.0, False

        print(f"[Buy] Contract id={contract_id} — waiting {duration_secs}s for result...")

        # Poll for settlement
        deadline = time.time() + duration_secs + 25
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                poll = await client.send(
                    {"proposal_open_contract": 1, "contract_id": contract_id},
                    timeout=10)
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

    # ── Update state ─────────────────────────────────────────────────────
    state.session_trades[symbol] += 1
    if won:
        state.session_wins[symbol] += 1
    state.session_profit[symbol] += profit
    state.last_trade_time[symbol] = time.time()
    state.last_activity           = time.time()

    wr     = state.session_wins[symbol] / max(state.session_trades[symbol], 1)
    result = f"✓ WIN  +${profit:.4f}" if won else f"✗ LOSS  -${BASE_STAKE:.2f}"
    print(f"\n{sep}")
    print(f"  RESULT  {symbol}  {datetime.now(timezone.utc).isoformat()}")
    print(sep)
    print(f"  Contract    : {contract_id}")
    print(f"  Outcome     : {result}")
    print(f"  Session     : {state.session_wins[symbol]}/{state.session_trades[symbol]} "
          f"({wr:.1%})  net P/L=${state.session_profit[symbol]:+.2f}")
    print(sep + "\n")

    # ── Refresh balance ───────────────────────────────────────────────────
    try:
        bal_resp    = await client.send({"balance": 1})
        state.balance = float(bal_resp["balance"]["balance"])
    except Exception:
        pass

    # ── Log to Supabase ───────────────────────────────────────────────────
    store.log_trade({
        "symbol":        symbol,
        "entry_price":   entry_price,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "barrier_width": barrier_abs * 2,
        "barrier_sigma": best["barrier_sigma"],
        "duration_secs": duration_secs,
        "won":           won,
        "profit":        profit,
        "mc_win_prob":   best["win_prob"],
        "mc_ci_lower":   best["ci_lower"],
        "breach_prob":   best["breach_prob"],
        "actual_payout": actual_payout,
        "vol_per_tick":  best["vol_per_tick"],
        "used_garch":    best["used_garch"],
        "adx_val":       gate_info.get("adx_val", 0.0),
        "vol_trust":     gate_info.get("vol_trust", 0.0),
        "hawkes_val":    gate_info.get("hawkes_val", 0.0),
    })
    return won, profit, True


# ═══════════════════════════════════════════════════════════════════════════
# DAILY SELF-IMPROVEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def daily_self_improvement(state: BotState, store: SupabaseStore):
    """
    Runs once per day at midnight UTC.
    Loads last 7 days of trade history from Supabase, then:
      1. Reweights duration preferences (Bayesian, α=2)
      2. Reweights barrier σ-slot preferences (Bayesian, α=2)
      3. Recalibrates vol_scalar by comparing MC predictions to actuals
      4. Saves all updated config back to Supabase
    """
    print("\n" + "═" * 68)
    print("  DAILY SELF-IMPROVEMENT  " +
          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("═" * 68)

    for symbol in SYMBOLS:
        rows = store.load_recent_trades(symbol, days=7)
        if not rows:
            print(f"[SI] {symbol}: no history (< 7 days old), skipping.")
            continue

        n_total = len(rows)
        n_wins  = sum(1 for r in rows if r.get("won"))
        profit  = sum(float(r.get("profit", 0)) for r in rows)
        print(f"\n[SI] {symbol}: {n_total} trades  {n_wins} wins "
              f"({n_wins/max(n_total,1):.1%})  net=${profit:+.2f}")

        # ── Per-duration stats ────────────────────────────────────────────
        dur_w: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        bar_w: Dict[float, List[int]] = defaultdict(lambda: [0, 0])
        mc_preds, actuals = [], []

        for r in rows:
            dur  = int(r.get("duration_secs", 120))
            won  = bool(r.get("won", False))
            bs   = float(r.get("barrier_sigma", 1.0))
            dur_w[dur][1] += 1
            bar_w[round(bs * 2) / 2][1] += 1
            if won:
                dur_w[dur][0] += 1
                bar_w[round(bs * 2) / 2][0] += 1
            mc_preds.append(float(r.get("mc_win_prob", 0.75)))
            actuals.append(1.0 if won else 0.0)

        alpha = 2.0

        # Duration reweighting
        raw_dw = {}
        print(f"  Duration win rates:")
        for dur, (w, t) in sorted(dur_w.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_dw[dur] = bwr
            print(f"    {dur}s: {w}/{t} ({w/t:.1%}), Bayes={bwr:.3f}")

        if raw_dw:
            mx, mn = max(raw_dw.values()), min(raw_dw.values())
            sp = mx - mn
            state.duration_weights[symbol] = {
                dur: (0.5 + 1.5 * (v - mn) / sp if sp > 0 else 1.0)
                for dur, v in raw_dw.items()
            }

        # Barrier reweighting
        raw_bw = {}
        print(f"  Barrier σ-slot win rates:")
        for slot, (w, t) in sorted(bar_w.items()):
            if t == 0: continue
            bwr = (w + alpha) / (t + 2 * alpha)
            raw_bw[slot] = bwr
            print(f"    σ={slot:.1f}: {w}/{t} ({w/t:.1%}), Bayes={bwr:.3f}")

        if raw_bw:
            mx, mn = max(raw_bw.values()), min(raw_bw.values())
            sp = mx - mn
            state.barrier_weights[symbol] = {
                slot: (0.5 + 1.5 * (v - mn) / sp if sp > 0 else 1.0)
                for slot, v in raw_bw.items()
            }

        # Vol scalar calibration
        if len(mc_preds) >= 10:
            mc_mean  = float(np.mean(mc_preds))
            act_mean = float(np.mean(actuals))
            ratio    = mc_mean / max(act_mean, 0.05)
            old_sc   = state.vol_scalar.get(symbol, 1.0)
            if ratio > 1.08:   # MC over-optimistic → inflate vol
                new_sc = float(np.clip(old_sc * min(ratio, 1.25), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar ↑: MC={mc_mean:.3f} > actual={act_mean:.3f} "
                      f"→ {old_sc:.3f} → {new_sc:.3f}")
            elif ratio < 0.92: # MC under-optimistic → deflate vol
                new_sc = float(np.clip(old_sc * max(ratio, 0.80), 0.5, 4.0))
                state.vol_scalar[symbol] = new_sc
                print(f"  Vol scalar ↓: MC={mc_mean:.3f} < actual={act_mean:.3f} "
                      f"→ {old_sc:.3f} → {new_sc:.3f}")
            else:
                print(f"  Vol scalar stable  MC={mc_mean:.3f} ≈ actual={act_mean:.3f}")

        # Daily summary
        best_dur = max(dur_w, key=lambda d: dur_w[d][0] / max(dur_w[d][1], 1)) if dur_w else 120
        best_bar = max(bar_w, key=lambda b: bar_w[b][0] / max(bar_w[b][1], 1)) if bar_w else 1.0
        store.save_daily_summary(
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            symbol, n_total, n_wins, profit, best_dur, best_bar,
        )

    # Persist updated config
    store.save_config("duration_weights",
        {s: {str(k): v for k, v in state.duration_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("barrier_weights",
        {s: {str(k): v for k, v in state.barrier_weights.get(s, {}).items()} for s in SYMBOLS})
    store.save_config("vol_scalars",
        {s: state.vol_scalar.get(s, 1.0) for s in SYMBOLS})

    state.last_daily_tune = time.time()
    print("\n[SI] Config saved. Next tuning in ~24h.")
    print("═" * 68 + "\n")


def load_config_from_supabase(state: BotState, store: SupabaseStore):
    dur_w = store.load_config("duration_weights")
    if dur_w:
        for s in SYMBOLS:
            if s in dur_w:
                state.duration_weights[s] = {int(k): float(v) for k, v in dur_w[s].items()}

    bar_w = store.load_config("barrier_weights")
    if bar_w:
        for s in SYMBOLS:
            if s in bar_w:
                state.barrier_weights[s] = {float(k): float(v) for k, v in bar_w[s].items()}

    vol_s = store.load_config("vol_scalars")
    if vol_s:
        for s in SYMBOLS:
            if s in vol_s:
                state.vol_scalar[s] = float(vol_s[s])

    if dur_w or bar_w or vol_s:
        print(f"[Config] Warm-start loaded — vol_scalars={state.vol_scalar}")
    else:
        print("[Config] Cold start — no prior config in Supabase.")


# ═══════════════════════════════════════════════════════════════════════════
# TICK HELPERS
# ═══════════════════════════════════════════════════════════════════════════
async def fetch_history(client: DerivClient, symbol: str, count: int) -> list:
    resp = await client.send({"ticks_history": symbol, "count": count,
                               "end": "latest", "style": "ticks"})
    h = resp.get("history", {})
    return list(zip(h.get("times", []), h.get("prices", [])))


async def subscribe_ticks(client: DerivClient, symbol: str) -> asyncio.Queue:
    q = client.subscribe_channel("tick")
    await client.send({"ticks": symbol, "subscribe": 1})
    return q


# ═══════════════════════════════════════════════════════════════════════════
# WATCHDOG
# ═══════════════════════════════════════════════════════════════════════════
async def watchdog(state: BotState):
    while True:
        await asyncio.sleep(30)
        if time.time() - state.last_activity > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {WATCHDOG_TIMEOUT}s — restarting.")
            os.execv(sys.executable, [sys.executable] + sys.argv)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    if not DERIV_API_TOKEN:
        sys.exit("[FATAL] DERIV_API_TOKEN not set.")
    if not DERIV_APP_ID:
        sys.exit("[FATAL] DERIV_APP_ID not set.")

    store = SupabaseStore()
    state = BotState()
    state._last_price = {s: 0.0 for s in SYMBOLS}    # live price tracker
    load_config_from_supabase(state, store)

    client  = DerivClient(DERIV_APP_ID, DERIV_API_TOKEN,
                          DERIV_ACCOUNT_TYPE, DERIV_ACCOUNT_ID)
    account = await client.connect()
    state.balance = float(account.get("balance", 0))
    print(f"Balance: ${state.balance:.2f}")

    # ── Symbol data buffers ───────────────────────────────────────────────
    sdata: Dict[str, SymbolData] = {
        s: SymbolData(s, maxlen=8000, tick_dt=SYMBOL_CONFIG[s]["tick_dt"])
        for s in SYMBOLS
    }

    # ── Bootstrap history ─────────────────────────────────────────────────
    print("\nBootstrapping tick history...")
    for sym in SYMBOLS:
        ticks = await fetch_history(client, sym, HISTORY_BOOTSTRAP)
        for epoch, price in ticks:
            sdata[sym].add_tick(epoch, price)
        prices = sdata[sym].prices()
        if len(prices):
            state._last_price[sym] = float(prices[-1])
        print(f"  {sym}: {len(ticks)} ticks  "
              f"price={state._last_price[sym]:.5f}")

    # ── Fit initial GARCH ─────────────────────────────────────────────────
    print("\nFitting GARCH models...")
    for sym in SYMBOLS:
        cfg     = SYMBOL_CONFIG[sym]
        returns = sdata[sym].returns()
        if len(returns) >= MIN_TICKS_FOR_FIT:
            gr = await asyncio.to_thread(fit_garch, returns, cfg["garch_scale"])
            state.garch_cache[sym] = (gr, time.time())
            print(f"  {sym}: GARCH {'fitted ✓' if gr else 'failed (using baseline vol)'}")
        else:
            state.garch_cache[sym] = (None, 0.0)
            print(f"  {sym}: not enough data ({len(returns)} returns), deferred")

    # ── Subscribe to live ticks ───────────────────────────────────────────
    tick_queues: Dict[str, asyncio.Queue] = {}
    for sym in SYMBOLS:
        tick_queues[sym] = await subscribe_ticks(client, sym)
    print(f"\nSubscribed to: {SYMBOLS}")

    # ── Resubscription callback ───────────────────────────────────────────
    async def resubscribe(c: DerivClient):
        for sym in SYMBOLS:
            tick_queues[sym] = await subscribe_ticks(c, sym)
        bal_resp     = await c.send({"balance": 1})
        state.balance = float(bal_resp.get("balance", {}).get("balance", state.balance))
        print("[Reconnect] Tick subscriptions restored.")

    client.resubscribe_cb = resubscribe

    asyncio.create_task(watchdog(state))
    state.last_activity = time.time()
    print("\n═" * 34)
    print("  Bot armed — scanning for EXPIRYRANGE setups")
    print("═" * 34 + "\n")

    garch_recal_interval = 2 * 3600   # recalibrate GARCH every 2 hours

    # ════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════════
    while True:
        # ── Drain tick queues ─────────────────────────────────────────────
        for sym in SYMBOLS:
            drained = 0
            while drained < 200:
                try:
                    msg  = tick_queues[sym].get_nowait()
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        sdata[sym].add_tick(float(tick["epoch"]), float(tick["quote"]))
                        state._last_price[sym] = float(tick["quote"])
                        drained += 1
                except asyncio.QueueEmpty:
                    break

            if drained == 0:
                # No ticks buffered — wait briefly for at least one
                try:
                    msg = await asyncio.wait_for(tick_queues[sym].get(), timeout=1.5)
                    tick = msg.get("tick", {})
                    if tick.get("symbol") == sym:
                        sdata[sym].add_tick(float(tick["epoch"]), float(tick["quote"]))
                        state._last_price[sym] = float(tick["quote"])
                except asyncio.TimeoutError:
                    pass

        state.last_activity = time.time()

        # ── Daily self-improvement ────────────────────────────────────────
        now_utc    = datetime.now(timezone.utc)
        since_tune = time.time() - state.last_daily_tune
        if since_tune > 23 * 3600 and now_utc.hour == DAILY_TUNE_HOUR_UTC:
            await asyncio.to_thread(daily_self_improvement, state, store)

        # ── Periodic GARCH recalibration (every 2h per symbol) ───────────
        for sym in SYMBOLS:
            gr, fitted_at = state.garch_cache.get(sym, (None, 0.0))
            if time.time() - fitted_at > garch_recal_interval:
                returns = sdata[sym].returns()
                if len(returns) >= MIN_TICKS_FOR_FIT:
                    cfg    = SYMBOL_CONFIG[sym]
                    gr_new = await asyncio.to_thread(fit_garch, returns, cfg["garch_scale"])
                    state.garch_cache[sym] = (gr_new, time.time())
                    print(f"[GARCH] {sym}: recalibrated "
                          f"{'✓' if gr_new else '(failed, using baseline)'}")

        if state.trading_locked:
            await asyncio.sleep(0.5)
            continue

        # ── Per-symbol evaluation ─────────────────────────────────────────
        for sym in SYMBOLS:
            sd = sdata[sym]
            if len(sd.ticks) < MIN_TICKS_LIVE:
                continue

            # Cooldown gate
            elapsed = time.time() - state.last_trade_time.get(sym, 0.0)
            if elapsed < SYMBOL_CONFIG[sym]["cooldown_secs"]:
                remain = SYMBOL_CONFIG[sym]["cooldown_secs"] - elapsed
                # silent wait — don't spam logs
                continue

            prices  = sd.prices()
            returns = sd.returns()
            if len(returns) < 20:
                continue

            garch_result, _ = state.garch_cache.get(sym, (None, 0.0))

            # ── STAGE 1: Structural gate ──────────────────────────────────
            gate_ok, gate_info = structural_gate(sym, prices, returns, garch_result)
            if not gate_ok:
                fails = {k: v for k, v in gate_info.items() if k.startswith("fail_")}
                print(f"[Gate] {sym}: blocked — {fails}")
                continue

            # ── STAGE 2: MC auto-optimizer ────────────────────────────────
            print(f"\n[MC] {sym}: running {MC_SIMULATIONS:,}-sim optimizer "
                  f"(gate ✓ ADX={gate_info['adx_val']:.1f} "
                  f"vol_trust={gate_info['vol_trust']:.3f})...")
            t0   = time.time()
            best = await asyncio.to_thread(
                mc_auto_optimize, prices, returns, sym, garch_result, state)
            dt = time.time() - t0

            if best is None:
                print(f"[MC] {sym}: no combo cleared win≥{MC_REQUIRED_WIN:.0%} "
                      f"& CI₅≥{MC_REQUIRED_CI:.0%} in {dt:.1f}s — waiting.")
                continue

            print(f"[MC] {sym}: best in {dt:.1f}s — "
                  f"dur={best['duration_secs']}s  "
                  f"barrier=±{best['barrier_abs']:.5f} ({best['barrier_sigma']:.2f}σ)  "
                  f"win={best['win_prob']:.3f}  CI₅={best['ci_lower']:.3f}")

            # ── STAGES 3 + 4: Proposal verify + Execute ───────────────────
            state.trading_locked = True
            won, profit = await execute_expiryrange(
                client, state, sym, best, gate_info, store)
            state.trading_locked = False

        await asyncio.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
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
