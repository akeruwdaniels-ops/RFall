"""
Deriv Multi-Symbol Rise/Fall Trading Bot - FULL POWER
========================================================
Single-file bot. Scans all eligible synthetic-index symbols (1HZ excluded),
runs a 12-layer intelligence pipeline per symbol using REAL fitted
statistical models (not heuristic approximations), fuses evidence into a
single directional probability via Bayesian log-odds combination, auto-
selects trade duration via Monte Carlo simulation conditioned on the fitted
models, and trades the single strongest signal at a time with a 1.24x /
3-step martingale and balance-scaled staking.

MODEL FITTING vs LIVE SCORING
------------------------------
Fitting HMM/GARCH/Hawkes/OU is computationally expensive, so it only happens
during calibration: once at startup (full universe), then every 2 hours
(top-K deep dive) or after 2 consecutive losses on a symbol (rate-limited,
that symbol's deep dive). Live trading between calibrations just evaluates
the cached fitted models against new ticks - cheap, fast, no refitting.

Symbols without a fitted model yet (before their first calibration) return
no signal and are simply not eligible for selection - this is automatic
and correct, no special-casing needed.

CONNECTION: new Deriv Options API (REST OTP bootstrap), verified against
developers.deriv.com as of 2026-06:
    REST  GET  /trading/v1/options/accounts            -> resolve account_id
    REST  POST /trading/v1/options/accounts/{id}/otp    -> pre-auth WS URL
    No `authorize` message needed - the OTP URL is already authenticated.
    OTP tokens are short-lived/single-use, so a fresh one is fetched on
    every (re)connect; the client auto-reconnects with backoff and replays
    subscriptions (balance + ticks for every symbol) after each reconnect.
    `active_symbols` no longer accepts `product_type`; its response field
    is `underlying_symbol` (not `symbol`). `contracts_for` no longer takes
    `currency`. Buy `parameters` now requires `underlying_symbol` (not
    `symbol`). Tick responses keep the `symbol` field unchanged.

ENV VARS REQUIRED:
    DERIV_APP_ID        - your app_id from a NEW developers.deriv.com application
                           (legacy app_ids, e.g. the old demo id 1089, do NOT
                           work with the new Options API)
    DERIV_API_TOKEN     - API token (personal access token) for your Deriv account
    DERIV_ACCOUNT_TYPE  - "demo" (default, safe) or "real". Picked explicitly
                           rather than guessed, so the bot never trades on
                           your real-money account by accident.
    DERIV_ACCOUNT_ID    - optional; skips the accounts lookup and uses this
                           account_id directly
"""

import asyncio
import json
import os
import random
import time
import math
import warnings
import numpy as np
import requests
import websockets
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict

from scipy.optimize import minimize
from scipy.stats import rankdata, norm
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn.hmm import GaussianHMM
from arch import arch_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG  (tune via your own walk-forward results before scaling up stakes)
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID") or None

# ── Connection (new Deriv Options API) ──
API_BASE = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH = "/trading/v1/options/accounts/{account_id}/otp"

MIN_STAKE = 0.35
STAKE_PCT = 0.02                       # stake = max(MIN_STAKE, balance * STAKE_PCT)

MARTINGALE_FACTOR = 1.24
MARTINGALE_MAX_STEPS = 3               # up to 3 recovery steps after the initial stake

SCHEDULED_CALIBRATION_INTERVAL = 2 * 60 * 60   # seconds (2 hours)
LOSS_TRIGGER_THRESHOLD = 2                     # consecutive losses on the SAME symbol
MAX_LOSS_CALIBRATIONS_PER_24H = 3              # rate limiter, default - tune as needed
CALIBRATION_COOLDOWN = 5 * 60                  # grace period after calibration ends
TOP_K_DEEP_DIVE = 5                            # symbols deep-validated per calibration
HISTORY_BOOTSTRAP_COUNT = 3000                 # ticks fetched per symbol at startup

CONFIDENCE_THRESHOLD = 0.11            # minimum ensemble score to trade (0-1 scale)
MIN_SCORE_GAP = 0.03                   # required gap over runner-up symbol
CANDIDATE_DURATIONS = [1, 3, 5, 10, 15]  # ticks, Monte Carlo picks the best of these
MC_SIMULATIONS = 500

MIN_TICKS_FOR_FIT = 200                # minimum ticks before a model can be fitted
MIN_TICKS_LIVE = 60                    # minimum ticks before live layers (Markov etc.) run


# ---------------------------------------------------------------------------
# SHARED STATE  (single source of truth - every module reads/writes through this)
# ---------------------------------------------------------------------------
class TradeState:
    def __init__(self):
        self.balance = 0.0
        self.trading_locked = False
        self.trade_in_progress = False
        self.consecutive_losses = defaultdict(int)
        self.reliability = defaultdict(lambda: 1.0)
        self.loss_triggered_calibrations_24h = deque()
        self.last_scheduled_calibration = time.time()
        self.last_calibration_end = 0.0
        self.model_cache: Dict[str, "SymbolModels"] = {}


@dataclass
class SymbolModels:
    fitted: bool = False
    fitted_at: float = 0.0
    origin_epoch: float = 0.0
    hmm_model: Optional[object] = None
    garch_result: Optional[object] = None
    garch_scale: float = 1000.0
    ou_params: Optional[dict] = None
    hawkes_up: Optional[dict] = None
    hawkes_up_events: Optional[np.ndarray] = None
    hawkes_down: Optional[dict] = None
    hawkes_down_events: Optional[np.ndarray] = None


class SymbolData:
    def __init__(self, symbol, maxlen=4000):
        self.symbol = symbol
        self.ticks = deque(maxlen=maxlen)  # (epoch, price)

    def add_tick(self, epoch, price):
        self.ticks.append((epoch, price))

    def prices(self):
        return np.array([p for _, p in self.ticks], dtype=float)

    def epochs(self):
        return np.array([e for e, _ in self.ticks], dtype=float)

    def returns(self):
        p = self.prices()
        if len(p) < 2:
            return np.array([])
        return np.diff(p) / p[:-1]

    def slice_copy(self, n):
        """Returns a new SymbolData containing only the first n ticks - used to
        build a clean train-set for walk-forward validation without mutating self."""
        new_sd = SymbolData(self.symbol, maxlen=n + 10)
        for e, p in list(self.ticks)[:n]:
            new_sd.add_tick(e, p)
        return new_sd


# ---------------------------------------------------------------------------
# DERIV API CLIENT - new Options API (REST OTP bootstrap, auto-reconnecting)
# ---------------------------------------------------------------------------
class DerivClient:
    """
    Client for the new Deriv Options API.

    Auth flow: REST GET .../accounts -> resolve account_id -> REST POST
    .../accounts/{id}/otp -> pre-authenticated WS URL. No `authorize`
    message is sent or needed; the OTP URL is already scoped to the account.

    OTP URLs are short-lived and single-use (per developers.deriv.com), so a
    fresh one is fetched on every connect AND every reconnect. After the
    first successful connect, this client auto-reconnects in the background
    with exponential backoff and calls `resubscribe_cb` (if set) so the
    caller can replay its balance/tick subscriptions.
    """

    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE = 2.0
    RECONNECT_CAP = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id = app_id
        self.token = token
        self.account_type = account_type
        self.account_id = account_id
        self.ws = None
        self.req_id = 0
        self.pending = {}
        self.subscriptions = defaultdict(list)  # msg_type -> list[asyncio.Queue]
        self.account = None
        self.resubscribe_cb = None  # async callable(client), replayed after reconnect
        self._running = False
        self._reader_task = None
        self._ka_task = None

    # ---- REST bootstrap ----
    def _rest_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json",
        }

    def _resolve_account_id_sync(self):
        url = f"{API_BASE}{ACCOUNTS_PATH}"
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
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
            f"No '{self.account_type}' account found via {ACCOUNTS_PATH}. "
            f"Set DERIV_ACCOUNT_ID explicitly, or create one first via "
            f"POST {ACCOUNTS_PATH}. Accounts returned: {data}"
        )

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        url = f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}"
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ---- connection lifecycle ----
    async def connect(self):
        """Connects once (raises on failure, so startup misconfiguration
        fails fast) then runs the supervisor loop forever in the background."""
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        # IMPORTANT: start the reader (and heartbeat) BEFORE sending anything.
        # `send()` blocks on a future that is only resolved by `_dispatch()`,
        # which only runs inside `_read_loop()`. If the reader isn't already
        # running, the balance handshake below times out forever (this was
        # the cause of a repeated TimeoutError/CancelledError crash loop).
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task = asyncio.create_task(self._heartbeat())
        bal = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(
            f"Connected ({self.account_type}). "
            f"loginid={self.account.get('loginid')} balance={self.account.get('balance')}"
        )

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[DerivClient] WS connection lost: {e}")

    async def _supervise(self):
        """Watches the current reader task; on disconnect, cleans up and
        reconnects with exponential backoff, restarting reader+heartbeat
        each time inside `_connect_once`."""
        while self._running:
            if self._reader_task is not None:
                await self._reader_task

            if self._ka_task is not None:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Deriv WS disconnected"))
            self.pending.clear()
            self.ws = None

            if not self._running:
                break

            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(
                    self.RECONNECT_BASE * (2 ** (attempt - 1)), self.RECONNECT_CAP
                ) + random.uniform(0, 1)
                print(f"[DerivClient] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[DerivClient] Reconnect attempt {attempt} failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id = data.get("req_id")
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



async def fetch_tradable_symbols(client):
    """Builds the symbol universe dynamically: synthetic indices only, 1HZ
    variants excluded, only symbols that actually support CALL/PUT contracts.

    New API: `active_symbols` response field is `underlying_symbol` (renamed
    from `symbol`), and `contracts_for` no longer takes `currency`."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[fetch_tradable_symbols] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" in symbol:
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)
    print(f"[fetch_tradable_symbols] {len(candidates)} synthetic-index candidates before contracts_for check")

    verified = []
    cf_errors = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                cf_errors.append(f"{symbol}: {cf['error']}")
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception as e:
            cf_errors.append(f"{symbol}: {type(e).__name__}: {e}")
        await asyncio.sleep(0.05)  # light throttle - avoid bursting dozens of requests at once

    if cf_errors:
        print(f"[fetch_tradable_symbols] {len(cf_errors)}/{len(candidates)} contracts_for calls failed, e.g.: {cf_errors[:3]}")
    return verified


async def fetch_history(client, symbol, count=HISTORY_BOOTSTRAP_COUNT):
    resp = await client.send({"ticks_history": symbol, "count": count, "end": "latest", "style": "ticks"})
    history = resp.get("history", {})
    times = history.get("times", [])
    prices = history.get("prices", [])
    return list(zip(times, prices))


async def buy_contract(client, symbol, direction, duration, duration_unit, stake):
    contract_type = "CALL" if direction > 0 else "PUT"
    req = {
        "buy": "1",
        "price": stake,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        },
    }
    resp = await client.send(req)
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", "buy failed"))
    return resp["buy"]["contract_id"]


async def wait_for_contract_result(client, contract_id):
    q = client.subscribe_channel("proposal_open_contract")
    await client.send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
    while True:
        data = await q.get()
        poc = data.get("proposal_open_contract", {})
        if poc.get("contract_id") == contract_id and poc.get("is_sold"):
            profit = float(poc.get("profit", 0))
            return profit > 0, profit


# ---------------------------------------------------------------------------
# LAYER 1: MARKOV CHAIN (order-2, Laplace/Dirichlet smoothed)
# ---------------------------------------------------------------------------
def markov_directional_prob(returns, order=2, alpha_smooth=1.0):
    signs = np.sign(returns)
    signs = signs[signs != 0]
    if len(signs) < order + 20:
        return 0.5
    table = defaultdict(lambda: [alpha_smooth, alpha_smooth])  # [down_count, up_count]
    for i in range(len(signs) - order):
        state = tuple(signs[i:i + order])
        idx = 1 if signs[i + order] > 0 else 0
        table[state][idx] += 1
    current_state = tuple(signs[-order:])
    down_c, up_c = table[current_state]
    return float(up_c / (up_c + down_c))


# ---------------------------------------------------------------------------
# LAYER 2: HIDDEN MARKOV MODEL (real Baum-Welch fit via hmmlearn)
# ---------------------------------------------------------------------------
def fit_hmm(returns, n_states=3):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        X = returns.reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=80, random_state=42)
        model.fit(X)
        return model
    except Exception as e:
        print(f"[HMM] fit failed: {e}")
        return None


def hmm_trend_weight(model, recent_returns):
    """Returns (trend_weight, directional_lean). trend_weight: how much to trust
    momentum layers right now (low variance dominant state = trending = trust
    momentum). directional_lean: HMM's own posterior-weighted directional vote."""
    if model is None or len(recent_returns) < 5:
        return 0.5, 0.0
    try:
        X = recent_returns.reshape(-1, 1)
        posteriors = model.predict_proba(X)
        current = posteriors[-1]
        means = model.means_.flatten()
        variances = np.array([np.sqrt(c[0][0]) for c in model.covars_])
        lean = float(np.sum(current * means))
        dominant = int(np.argmax(current))
        vol_rank = variances[dominant] / (np.max(variances) + 1e-9)
        trend_weight = float(np.clip(1.0 - vol_rank, 0.2, 0.85))
        lean_signal = float(np.tanh(lean * 200))
        return trend_weight, lean_signal
    except Exception:
        return 0.5, 0.0


# ---------------------------------------------------------------------------
# LAYER 3: HAWKES PROCESS (real exponential-kernel MLE fit via scipy)
# ---------------------------------------------------------------------------
def hawkes_negloglik(params, event_times, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e10
    ll = -mu * T
    A = 0.0
    last_t = 0.0
    for i, ti in enumerate(event_times):
        if i > 0:
            A = math.exp(-beta * (ti - last_t)) * (1 + A)
        lam = mu + alpha * A
        if lam <= 0:
            return 1e10
        ll += math.log(lam)
        last_t = ti
    comp = (alpha / beta) * np.sum(1 - np.exp(-beta * (T - event_times)))
    ll -= comp
    return -ll


def fit_hawkes(event_times, T):
    if len(event_times) < 10 or T <= 0:
        return None
    init = [max(len(event_times) / T * 0.5, 1e-4), 0.3, 1.0]
    try:
        res = minimize(
            hawkes_negloglik, init, args=(event_times, T),
            bounds=[(1e-6, None), (0.0, None), (1e-6, None)],
            method="L-BFGS-B",
        )
        if not res.success:
            return None
        mu, alpha, beta = res.x
        if alpha >= beta:
            return None
        return {"mu": mu, "alpha": alpha, "beta": beta}
    except Exception as e:
        print(f"[Hawkes] fit failed: {e}")
        return None


def hawkes_intensity_now(params, event_times, current_t):
    if params is None or event_times is None or len(event_times) == 0:
        return 0.0
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    past = event_times[event_times <= current_t]
    if len(past) == 0:
        return mu
    excitation = np.sum(alpha * np.exp(-beta * (current_t - past)))
    return float(mu + excitation)


def fit_symbol_hawkes(sd):
    returns = sd.returns()
    epochs = sd.epochs()
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None, None, None, None, None
    thresh = 0.5 * np.std(returns) if np.std(returns) > 0 else 1e-9
    origin = epochs[0]
    event_epochs = epochs[1:]
    up_times = (event_epochs[returns > thresh] - origin).astype(float)
    down_times = (event_epochs[returns < -thresh] - origin).astype(float)
    T = float(epochs[-1] - origin)
    hawkes_up = fit_hawkes(up_times, T) if len(up_times) >= 10 else None
    hawkes_down = fit_hawkes(down_times, T) if len(down_times) >= 10 else None
    return origin, hawkes_up, up_times, hawkes_down, down_times


# ---------------------------------------------------------------------------
# LAYER 4: ORNSTEIN-UHLENBECK (OLS / Vasicek-style calibration)
# ---------------------------------------------------------------------------
def fit_ou(prices, dt=1.0):
    if len(prices) < 30:
        return None
    x, y = prices[:-1], prices[1:]
    try:
        b, a = np.polyfit(x, y, 1)
    except Exception:
        return None
    b = float(np.clip(b, 1e-6, 0.999999))
    theta = -math.log(b) / dt
    mu = a / (1 - b)
    resid = y - (a + b * x)
    resid_var = np.var(resid)
    denom = 1 - b ** 2
    sigma = math.sqrt(resid_var * 2 * theta / denom) if denom > 1e-9 else math.sqrt(max(resid_var, 1e-12))
    return {"theta": theta, "mu": mu, "sigma": sigma}


def ou_reversion_signal(prices, ou_params):
    if ou_params is None or len(prices) < 2:
        return {"z": 0.0, "reversion_dir": 0.0, "strength": 0.0}
    mu, sigma = ou_params["mu"], (ou_params["sigma"] if ou_params["sigma"] > 0 else 1e-9)
    z = (prices[-1] - mu) / sigma
    theta_norm = float(np.clip(ou_params["theta"], 0, 5) / 5)
    strength = float(np.clip(abs(z) / 2 * theta_norm, 0, 1))
    return {"z": float(z), "reversion_dir": float(-np.sign(z)), "strength": strength}


# ---------------------------------------------------------------------------
# LAYER 5: HURST EXPONENT (real rescaled-range / R-S analysis)
# ---------------------------------------------------------------------------
def hurst_rs(prices, min_window=10):
    prices = np.asarray(prices)
    n = len(prices)
    if n < 100:
        return 0.5
    max_window = n // 2
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int))
    rs_points = []
    for w in window_sizes:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = prices[i * w:(i + 1) * w]
            mean = np.mean(chunk)
            dev = np.cumsum(chunk - mean)
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_points.append((w, np.mean(rs_chunk)))
    if len(rs_points) < 3:
        return 0.5
    log_w = np.log([w for w, _ in rs_points])
    log_rs = np.log([rs for _, rs in rs_points])
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LAYER 6: ARFIMA-STYLE LONG MEMORY (fractional differencing + AR(1))
# ---------------------------------------------------------------------------
def fractional_diff_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def arfima_bias(returns, hurst, lookback=150):
    if len(returns) < 60:
        return 0.0
    d = float(np.clip(hurst - 0.5, -0.49, 0.49))
    recent = returns[-lookback:]
    n = len(recent)
    w = fractional_diff_weights(d, n)
    diff_series = np.convolve(recent, w, mode="valid")
    if len(diff_series) < 15:
        return float(np.tanh(diff_series[-1] * 50)) if len(diff_series) else 0.0
    try:
        ar_model = AutoReg(diff_series, lags=1, old_names=False).fit()
        forecast = ar_model.predict(start=len(diff_series), end=len(diff_series)).iloc[0]
    except Exception:
        forecast = diff_series[-1]
    return float(np.tanh(forecast * 50))


# ---------------------------------------------------------------------------
# LAYER 7: GARCH(1,1) (real MLE fit via the `arch` package)
# ---------------------------------------------------------------------------
def fit_garch(returns, scale=1000.0):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * scale
        am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        return am.fit(disp="off")
    except Exception as e:
        print(f"[GARCH] fit failed: {e}")
        return None


def garch_vol_trust(garch_result, returns, scale=1000.0):
    if garch_result is None:
        return 0.5, None
    try:
        forecast = garch_result.forecast(horizon=1, reindex=False)
        cond_vol = math.sqrt(float(forecast.variance.values[-1, 0])) / scale
        baseline_vol = np.std(returns) if np.std(returns) > 0 else 1e-9
        ratio = cond_vol / baseline_vol
        trust = 1.0 / (1.0 + max(ratio - 1, 0) * 2)
        return float(np.clip(trust, 0.1, 1.0)), cond_vol
    except Exception:
        return 0.5, None


# ---------------------------------------------------------------------------
# LAYER 8: SAMPLE ENTROPY (proper formula, not histogram Shannon entropy)
# ---------------------------------------------------------------------------
def sample_entropy_trust(returns, m=2, r_mult=0.2):
    if len(returns) < 30:
        return 0.5
    r = r_mult * np.std(returns)
    if r <= 0:
        return 0.5
    n = len(returns)

    def _phi(mm):
        x = np.array([returns[i:i + mm] for i in range(n - mm + 1)])
        count, total = 0, 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            count += np.sum(dist <= r) - 1
            total += len(x) - 1
        return count / total if total > 0 else 0.0

    phi_m, phi_m1 = _phi(m), _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return 0.5
    sampen = -math.log(phi_m1 / phi_m)
    return float(np.clip(1.0 / (1.0 + sampen), 0.1, 1.0))


# ---------------------------------------------------------------------------
# LAYER 9: KALMAN FILTER (real 2-state local-level + trend filter)
# ---------------------------------------------------------------------------
def kalman_trend_filter(prices, q_level=1e-5, q_trend=1e-6, r_obs=0.01):
    if len(prices) < 5:
        return 0.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_trend]])
    R = np.array([[r_obs]])
    x = np.array([[prices[0]], [0.0]])
    P = np.eye(2)
    for price in prices[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        y = price - (H @ x)[0, 0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / S
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P
    trend = x[1, 0]
    denom = np.std(prices) + 1e-9
    return float(np.clip(np.sign(trend) * min(abs(trend) / denom * 10, 1.0), -1, 1))


# ---------------------------------------------------------------------------
# LAYER 10: COPULA (real Gaussian copula via rank-normal transform)
# ---------------------------------------------------------------------------
def copula_agreement(symbol, returns_window_dict):
    symbols = list(returns_window_dict.keys())
    if symbol not in symbols or len(symbols) < 2:
        return 0.5
    min_len = min(len(v) for v in returns_window_dict.values())
    if min_len < 30:
        return 0.5
    data = np.array([returns_window_dict[s][-min_len:] for s in symbols]).T
    ranks = np.apply_along_axis(rankdata, 0, data) / (min_len + 1)
    normal_scores = norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4))
    corr = np.corrcoef(normal_scores.T)
    idx = symbols.index(symbol)
    target_sign = np.sign(normal_scores[-1, idx])
    weighted_agree, total_weight = 0.0, 0.0
    for j in range(len(symbols)):
        if j == idx:
            continue
        rho = abs(corr[idx, j])
        peer_sign = np.sign(normal_scores[-1, j])
        weighted_agree += rho * (1.0 if peer_sign == target_sign else 0.0)
        total_weight += rho
    if total_weight == 0:
        return 0.5
    return float(np.clip(weighted_agree / total_weight, 0, 1))


# ---------------------------------------------------------------------------
# MODEL FITTING ORCHESTRATOR (runs only during calibration)
# ---------------------------------------------------------------------------
def fit_symbol_models(sd) -> SymbolModels:
    models = SymbolModels()
    returns = sd.returns()
    prices = sd.prices()
    if len(returns) < MIN_TICKS_FOR_FIT:
        return models

    models.hmm_model = fit_hmm(returns)
    models.garch_result = fit_garch(returns, scale=models.garch_scale)
    models.ou_params = fit_ou(prices)
    origin, h_up, up_ev, h_down, down_ev = fit_symbol_hawkes(sd)
    models.origin_epoch = origin if origin is not None else sd.epochs()[0]
    models.hawkes_up, models.hawkes_up_events = h_up, up_ev
    models.hawkes_down, models.hawkes_down_events = h_down, down_ev

    models.fitted_at = time.time()
    models.fitted = True
    return models


# ---------------------------------------------------------------------------
# LAYER 11: BAYESIAN FUSION (log-odds evidence combination - owns final direction)
# ---------------------------------------------------------------------------
def compute_features(sd, models, returns_window_dict):
    """Evaluates all layers using the CACHED fitted models. Returns None if no
    model has been fitted yet for this symbol (it simply isn't tradable until
    its first calibration completes)."""
    if models is None or not models.fitted:
        return None
    returns = sd.returns()
    prices = sd.prices()
    if len(returns) < MIN_TICKS_LIVE:
        return None

    recent_returns = returns[-50:] if len(returns) >= 50 else returns
    trend_weight, hmm_lean = hmm_trend_weight(models.hmm_model, recent_returns)
    vol_trust, cond_vol = garch_vol_trust(models.garch_result, returns, models.garch_scale)
    ou = ou_reversion_signal(prices, models.ou_params)

    current_t = float(sd.epochs()[-1] - models.origin_epoch)
    lam_up = hawkes_intensity_now(models.hawkes_up, models.hawkes_up_events, current_t)
    lam_down = hawkes_intensity_now(models.hawkes_down, models.hawkes_down_events, current_t)
    hawkes_signal = (lam_up - lam_down) / (lam_up + lam_down + 1e-9)

    h = hurst_rs(prices)
    arfima = arfima_bias(returns, h)
    markov_p = markov_directional_prob(returns)
    kalman = kalman_trend_filter(prices)
    ent_trust = sample_entropy_trust(returns[-150:] if len(returns) >= 150 else returns)
    copula = copula_agreement(sd.symbol, returns_window_dict)

    return {
        "markov_p": markov_p,
        "hmm_lean": hmm_lean,
        "trend_weight": trend_weight,
        "hawkes": hawkes_signal,
        "ou_dir": ou["reversion_dir"],
        "ou_strength": ou["strength"],
        "hurst": h,
        "arfima_bias": arfima,
        "vol_trust": vol_trust,
        "entropy_trust": ent_trust,
        "kalman": kalman,
        "copula_agree": copula,
        "cond_vol": cond_vol,
        "ou_params": models.ou_params,
    }


def bayesian_fusion(features):
    """Each layer contributes a log-odds update weighted by its own relevance
    and the overall trust multiplier (vol_trust * entropy_trust). Summing
    log-odds and passing through a sigmoid is the standard Bayesian/naive-Bayes
    evidence-combination approach (assumes conditional independence of evidence
    sources - a simplifying assumption, noted here deliberately)."""
    p_markov = float(np.clip(features["markov_p"], 1e-3, 1 - 1e-3))
    markov_logit = math.log(p_markov / (1 - p_markov))

    trend_w = features["trend_weight"]
    evidence = [
        (markov_logit, 1.0),
        (features["hmm_lean"] * 2.0, trend_w),
        (features["hawkes"] * 2.5, trend_w),
        (features["ou_dir"] * features["ou_strength"] * 2.0, 1 - trend_w),
        (features["arfima_bias"] * 1.5, 0.6),
        (features["kalman"] * 1.5, 0.7),
        ((features["copula_agree"] - 0.5) * 2.0, 0.5),
    ]

    total_trust = features["vol_trust"] * features["entropy_trust"]
    log_odds, total_weight = 0.0, 0.0
    for log_ratio, weight in evidence:
        w = weight * total_trust
        log_odds += log_ratio * w
        total_weight += w

    p_up = float(np.clip(1.0 / (1.0 + math.exp(-log_odds)), 0.01, 0.99))
    confidence = abs(p_up - 0.5) * 2 * total_trust
    return p_up, confidence


# ---------------------------------------------------------------------------
# LAYER 12: MONTE CARLO DURATION SELECTOR
# ---------------------------------------------------------------------------
def monte_carlo_duration(prices, returns, direction, feats, candidate_durations, n_sims=MC_SIMULATIONS):
    """Takes the direction already decided by the Bayesian layer (does NOT
    re-decide direction) and simulates forward paths - using the GARCH
    conditional volatility and OU mean-reversion pull from the fitted models -
    to find which duration maximizes expected win probability."""
    if len(returns) < 20:
        return candidate_durations[0], 0.5

    cond_vol = feats.get("cond_vol")
    vol = cond_vol if cond_vol and cond_vol > 0 else (np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns))
    vol = vol if vol > 0 else 1e-6

    hawkes_signal = feats.get("hawkes", 0.0)
    drift = direction * abs(np.mean(returns[-50:])) * (1 + abs(hawkes_signal) * 0.5) if len(returns) >= 50 else 0.0

    ou_params = feats.get("ou_params")
    current_price = prices[-1]
    reversion_pull = 0.0
    if ou_params and ou_params.get("theta", 0) > 0:
        reversion_pull = ou_params["theta"] * (ou_params["mu"] - current_price) * 0.01

    best = None
    for dur in candidate_durations:
        steps = np.random.normal(drift + reversion_pull, vol, size=(n_sims, dur))
        path_totals = np.sum(steps, axis=1)
        wins = np.sum(path_totals > 0) if direction > 0 else np.sum(path_totals < 0)
        win_rate = wins / n_sims
        if best is None or win_rate > best[1]:
            best = (dur, win_rate)
    return best


# ---------------------------------------------------------------------------
# ENSEMBLE SELECTOR
# ---------------------------------------------------------------------------
def select_trade(symbol_scores, reliability):
    scored = []
    for symbol, (p_up, confidence) in symbol_scores.items():
        score = confidence * reliability.get(symbol, 1.0)
        direction = 1 if p_up > 0.5 else -1
        scored.append((symbol, direction, p_up, score))
    if not scored:
        return None
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[0]
    if top[3] < CONFIDENCE_THRESHOLD:
        return None
    if len(scored) > 1 and (top[3] - scored[1][3]) < MIN_SCORE_GAP:
        return None
    return top


# ---------------------------------------------------------------------------
# STAKING
# ---------------------------------------------------------------------------
def calculate_stake(balance):
    """stake = max($0.35, 2% of balance) - single formula, no seam/discontinuity."""
    return round(max(MIN_STAKE, balance * STAKE_PCT), 2)


def martingale_stakes(base_stake):
    stakes = [round(base_stake, 2)]
    for _ in range(MARTINGALE_MAX_STEPS):
        stakes.append(round(stakes[-1] * MARTINGALE_FACTOR, 2))
    return stakes


# ---------------------------------------------------------------------------
# TRADE EXECUTION
# ---------------------------------------------------------------------------
def log_trade(symbol, direction, stake, won, profit, step):
    ts = datetime.utcnow().isoformat()
    side = "CALL" if direction > 0 else "PUT"
    print(f"[{ts}] {symbol} {side} step={step} stake={stake:.2f} won={won} profit={profit:.2f}")


async def execute_sequence(client, state, symbol, direction, duration):
    state.trade_in_progress = True
    base_stake = calculate_stake(state.balance)
    stakes = martingale_stakes(base_stake)
    sequence_won = False
    try:
        for step, stake in enumerate(stakes):
            contract_id = await buy_contract(client, symbol, direction, duration, "t", stake)
            won, profit = await wait_for_contract_result(client, contract_id)
            log_trade(symbol, direction, stake, won, profit, step)
            if won:
                sequence_won = True
                break
    except Exception as e:
        print(f"Trade error on {symbol}: {e}")

    state.consecutive_losses[symbol] = 0 if sequence_won else state.consecutive_losses[symbol] + 1

    try:
        bal_resp = await client.send({"balance": 1})
        state.balance = bal_resp["balance"]["balance"]
    except Exception:
        pass

    state.trade_in_progress = False


# ---------------------------------------------------------------------------
# SYMBOL CALIBRATOR (trigger manager + FULL-POWER calibration engine)
# ---------------------------------------------------------------------------
def check_calibration_triggers(state):
    now = time.time()
    if now - state.last_calibration_end < CALIBRATION_COOLDOWN:
        return None
    if now - state.last_scheduled_calibration >= SCHEDULED_CALIBRATION_INTERVAL:
        return "scheduled", None
    for symbol, count in list(state.consecutive_losses.items()):
        if count >= LOSS_TRIGGER_THRESHOLD:
            recent = [t for t in state.loss_triggered_calibrations_24h if now - t < 86400]
            state.loss_triggered_calibrations_24h = deque(recent)
            if len(recent) < MAX_LOSS_CALIBRATIONS_PER_24H:
                return "loss_triggered", symbol
    return None


def walk_forward_validate(sd, train_frac=0.8, horizon=5, step=5):
    """REAL walk-forward validation: fit models on the first train_frac of the
    buffered ticks only, then step through the held-out remainder tick by tick
    (simulating live arrival), generating predictions from the FROZEN trained
    models and comparing to realized direction `horizon` ticks later. Returns
    (hit_rate, fitted_models) - the same models get cached for live trading if
    validation passes a sane bar."""
    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT + 100:
        return 0.5, None

    split = max(MIN_TICKS_FOR_FIT, int(n_ticks * train_frac))
    train_sd = sd.slice_copy(split)
    models = fit_symbol_models(train_sd)
    if not models.fitted:
        return 0.5, None

    eval_sd = sd.slice_copy(split)
    remaining_ticks = list(sd.ticks)[split:]
    hits, total = 0, 0
    for i in range(0, len(remaining_ticks) - horizon, step):
        eval_sd.add_tick(*remaining_ticks[i])
        feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
        if feats is None:
            continue
        p_up, _ = bayesian_fusion(feats)
        predicted_dir = 1 if p_up > 0.5 else -1
        current_price = remaining_ticks[i][1]
        future_price = remaining_ticks[i + horizon][1]
        actual_dir = 1 if future_price > current_price else -1
        hits += int(predicted_dir == actual_dir)
        total += 1

    hit_rate = hits / total if total > 0 else 0.5
    return hit_rate, models


async def run_calibration(state, symbol_data, symbols, trigger_reason):
    state.trading_locked = True
    kind, loss_symbol = trigger_reason
    start = time.time()
    print(f"[Calibrator] starting (trigger={kind}{':' + loss_symbol if loss_symbol else ''}). Trading locked.")

    if kind == "loss_triggered":
        state.loss_triggered_calibrations_24h.append(start)

    if kind == "initial":
        candidates = symbols
    else:
        scan_scores = {}
        for s in symbols:
            sd = symbol_data[s]
            if len(sd.ticks) < MIN_TICKS_LIVE:
                continue
            returns = sd.returns()
            scan_scores[s] = sample_entropy_trust(returns[-150:] if len(returns) >= 150 else returns)
        candidates = sorted(scan_scores, key=scan_scores.get, reverse=True)[:TOP_K_DEEP_DIVE]
        if loss_symbol and loss_symbol not in candidates:
            candidates.append(loss_symbol)

    for s in candidates:
        sd = symbol_data[s]
        if len(sd.ticks) < MIN_TICKS_FOR_FIT + 100:
            print(f"[Calibrator] {s}: not enough ticks yet, skipping this cycle.")
            continue
        hit_rate, models = walk_forward_validate(sd)
        if models is not None:
            state.model_cache[s] = models
        state.reliability[s] = float(np.clip(hit_rate / 0.5, 0.3, 1.5))
        state.consecutive_losses[s] = 0
        print(f"[Calibrator] {s}: walk-forward hit_rate={hit_rate:.3f} reliability={state.reliability[s]:.2f}")

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end = time.time()
    print(f"[Calibrator] complete in {state.last_calibration_end - start:.1f}s. Updated: {candidates}")
    state.trading_locked = False


# ---------------------------------------------------------------------------
# STREAM CONSUMERS
# ---------------------------------------------------------------------------
async def tick_consumer(queue, symbol_data):
    while True:
        data = await queue.get()
        tick = data.get("tick")
        if not tick:
            continue
        symbol = tick.get("symbol")
        if symbol in symbol_data:
            symbol_data[symbol].add_tick(tick["epoch"], tick["quote"])


async def balance_consumer(queue, state):
    while True:
        data = await queue.get()
        bal = data.get("balance")
        if bal:
            state.balance = bal["balance"]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    if not DERIV_API_TOKEN:
        raise RuntimeError("Set the DERIV_API_TOKEN environment variable.")
    if not DERIV_APP_ID:
        raise RuntimeError(
            "Set the DERIV_APP_ID environment variable to your app_id from "
            "developers.deriv.com. Legacy app_ids (e.g. the old demo id "
            "1089) do NOT work with the new Options API."
        )
    if DERIV_ACCOUNT_TYPE not in ("demo", "real"):
        raise RuntimeError("DERIV_ACCOUNT_TYPE must be 'demo' or 'real'.")
    if DERIV_ACCOUNT_TYPE == "real":
        print("!" * 72)
        print("! DERIV_ACCOUNT_TYPE=real - this bot will trade with REAL MONEY.    !")
        print("! Set DERIV_ACCOUNT_TYPE=demo (or unset it) to use a demo account.  !")
        print("!" * 72)

    client = DerivClient(
        DERIV_APP_ID, DERIV_API_TOKEN,
        account_type=DERIV_ACCOUNT_TYPE, account_id=DERIV_ACCOUNT_ID,
    )
    account = await client.connect()
    print(f"Authorized as {account.get('loginid')}")

    state = TradeState()
    state.balance = account.get("balance", 0.0)
    print(f"Starting balance: {state.balance}")

    symbols = []
    for attempt in range(1, 6):
        symbols = await fetch_tradable_symbols(client)
        if symbols:
            break
        print(f"[main] No tradable symbols on attempt {attempt}/5, retrying in 3s...")
        await asyncio.sleep(3)
    if not symbols:
        raise RuntimeError("No tradable rise/fall symbols found (check API credentials/connectivity).")
    print(f"Tradable universe ({len(symbols)} symbols, 1HZ excluded): {symbols}")

    symbol_data = {s: SymbolData(s) for s in symbols}
    print("Bootstrapping tick history for all symbols (this funds the initial calibration)...")
    for s in symbols:
        history = await fetch_history(client, s)
        for epoch, price in history:
            symbol_data[s].add_tick(epoch, price)
        print(f"  {s}: {len(symbol_data[s].ticks)} ticks loaded")

    tick_queue = client.subscribe_channel("tick")
    balance_queue = client.subscribe_channel("balance")

    async def subscribe_all(c):
        """Replays balance + per-symbol tick subscriptions. Used for the
        initial subscribe and re-run as `resubscribe_cb` after every
        reconnect (a fresh OTP session has no memory of prior subscriptions)."""
        await c.send({"balance": 1, "subscribe": 1})
        for s in symbols:
            await c.send({"ticks": s, "subscribe": 1})

    client.resubscribe_cb = subscribe_all
    await subscribe_all(client)

    asyncio.create_task(tick_consumer(tick_queue, symbol_data))
    asyncio.create_task(balance_consumer(balance_queue, state))

    print("Running initial full-power calibration across the entire universe before trading begins...")
    await run_calibration(state, symbol_data, symbols, ("initial", None))

    print("Bot running. Entering main decision loop.")
    while True:
        await asyncio.sleep(2)

        if state.trading_locked or state.trade_in_progress:
            continue

        trigger = check_calibration_triggers(state)
        if trigger:
            await run_calibration(state, symbol_data, symbols, trigger)
            continue

        ready_symbols = [s for s in symbols if s in state.model_cache and len(symbol_data[s].ticks) >= MIN_TICKS_LIVE]
        if not ready_symbols:
            continue

        returns_window_dict = {s: symbol_data[s].returns()[-200:] for s in ready_symbols}

        symbol_scores = {}
        for s in ready_symbols:
            sd = symbol_data[s]
            feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
            if feats is None:
                continue
            p_up, confidence = bayesian_fusion(feats)
            symbol_scores[s] = (p_up, confidence)

        pick = select_trade(symbol_scores, state.reliability)
        if not pick:
            continue

        symbol, direction, p_up, score = pick
        sd = symbol_data[symbol]
        feats = compute_features(sd, state.model_cache.get(symbol), returns_window_dict)
        duration, exp_win_rate = monte_carlo_duration(
            sd.prices(), sd.returns(), direction, feats, CANDIDATE_DURATIONS
        )
        print(
            f"Selected {symbol} dir={'UP' if direction > 0 else 'DOWN'} "
            f"p_up={p_up:.3f} score={score:.3f} duration={duration}t exp_win={exp_win_rate:.2f}"
        )
        await execute_sequence(client, state, symbol, direction, duration)


if __name__ == "__main__":
    asyncio.run(main())
