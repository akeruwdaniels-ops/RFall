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
import io
import json
import os
import random
import sys
import time
import math
import contextlib
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

CONFIDENCE_THRESHOLD_DEFAULT = 0.11    # fallback only - real threshold is set adaptively by
                                        # the calibrator from the observed score distribution
                                        # (see ADAPTIVE_THRESHOLD_PERCENTILE below)
MIN_SCORE_GAP = 0.03                   # required gap over runner-up symbol
CANDIDATE_DURATIONS = [1, 3, 5, 7, 10]   # ticks - Deriv rejects tick contracts outside 1-10,
                                          # this was the cause of the repeated "Number of ticks
                                          # must be between 1 and 10" trade errors
MC_SIMULATIONS = 500
MIN_EXP_WIN_RATE = 0.45                # Monte Carlo sanity gate: if even the BEST candidate
                                        # duration's simulated win rate is below this, skip the
                                        # trade entirely rather than firing on a duration the
                                        # model itself thinks is a coin-flip-or-worse

ADAPTIVE_THRESHOLD_PERCENTILE = 65     # calibration sets CONFIDENCE_THRESHOLD to this
                                        # percentile of the actual confidence scores seen during
                                        # walk-forward replay, so the bar is empirically reachable
                                        # for the real data instead of a hand-picked guess

WATCHDOG_TIMEOUT = 5 * 60              # seconds of total silence (no tick, no loop iteration)
                                        # before the bot force-restarts itself in place
WATCHDOG_CHECK_INTERVAL = 20           # how often the watchdog checks for staleness

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
        self.last_activity = time.time()

        # Threshold: per-symbol, derived from each symbol's own OOS confidence
        # distribution during deep calibration. Falls back to global default
        # only for symbols not yet calibrated.
        self.adaptive_threshold = CONFIDENCE_THRESHOLD_DEFAULT   # global fallback
        self.per_symbol_threshold: Dict[str, float] = {}

        # Martingale recovery context — saved between main-loop iterations so
        # each recovery step waits for a genuine signal, not an instant re-entry
        self.recovery_symbol    = None
        self.recovery_step      = 0
        self.recovery_stake     = 0.0
        self.recovery_direction = 0

        # Step-0 (raw signal, no martingale recovery) win-rate tracking —
        # the only metric that honestly reveals whether the signal has edge
        self.step0_wins   = defaultdict(int)
        self.step0_total  = defaultdict(int)

        # Sequence accumulator — tracks stakes/profits across martingale steps
        # so log_trade_summary has the full picture when the sequence closes
        self.seq_stakes    = []    # stake placed at each step
        self.seq_profits   = []    # profit (negative = loss) at each step
        self.seq_balance_before = 0.0   # balance at sequence open
        self.seq_p_up      = 0.5
        self.seq_confidence= 0.0
        self.seq_duration  = 0


@dataclass
class SymbolModels:
    fitted: bool = False
    fitted_at: float = 0.0
    origin_epoch: float = 0.0
    tick_dt: float = 2.0             # actual measured dt at fit time, carried for re-use
    hmm_model: Optional[object] = None
    garch_result: Optional[object] = None
    garch_scale: float = 1000.0
    ou_params: Optional[dict] = None
    hawkes_up: Optional[dict] = None
    hawkes_up_events: Optional[np.ndarray] = None
    hawkes_down: Optional[dict] = None
    hawkes_down_events: Optional[np.ndarray] = None
    # per-layer fusion weights learned from OOS correlation during deep calibration
    # None means fall back to static defaults inside bayesian_fusion()
    per_layer_weights: Optional[dict] = None


class SymbolData:
    def __init__(self, symbol, maxlen=4000, tick_dt=2.0):
        self.symbol = symbol
        self.tick_dt = tick_dt          # seconds per tick: 1.0 for 1HZ, ~2.0 for R_
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

    def mean_tick_dt(self):
        """Compute actual mean inter-tick gap in seconds from the buffered epochs.
        Used to verify the tick_dt assumption and for activity ranking."""
        e = self.epochs()
        if len(e) < 2:
            return self.tick_dt
        return float(np.mean(np.diff(e)))

    def slice_copy(self, n):
        """Returns a new SymbolData containing only the first n ticks, carrying
        tick_dt through so re-fitted models use the correct rate."""
        new_sd = SymbolData(self.symbol, maxlen=n + 10, tick_dt=self.tick_dt)
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
    """Fetches R_ volatility indices only (R_10/25/50/75/100).
    Returns a list of verified CALL/PUT-eligible symbol names.
    1HZ symbols are handled separately by select_top_1hz()."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[fetch_tradable_symbols] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" in symbol:
            continue
        if not symbol.startswith("R_"):
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)
    print(f"[fetch_tradable_symbols] {len(candidates)} R_ candidates before contracts_for check")

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
        await asyncio.sleep(0.05)

    if cf_errors:
        print(f"[fetch_tradable_symbols] {len(cf_errors)} contracts_for calls failed, e.g.: {cf_errors[:3]}")
    print(f"[fetch_tradable_symbols] verified R_ symbols: {verified}")
    return verified


async def select_top_1hz(client, n_top=3):
    """Fetches all 1HZ synthetic-index symbols that support CALL/PUT, bootstraps
    a short tick history for each, then ranks by tick-flow consistency (lowest
    coefficient-of-variation of inter-tick gaps = most active / most liquid).
    Returns the top n_top as a list of symbol names.

    Why consistency rather than just speed: all 1HZ symbols nominally tick every
    second, but some have gaps and bursts (irregular flow) while others tick very
    evenly. Even gap distribution means more reliable statistical model fitting
    and more predictable execution timing."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[select_top_1hz] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" not in symbol:
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)

    print(f"[select_top_1hz] {len(candidates)} 1HZ candidates found: {candidates}")

    # verify CALL/PUT support
    verified = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception:
            continue
        await asyncio.sleep(0.05)

    print(f"[select_top_1hz] {len(verified)} CALL/PUT-eligible 1HZ symbols: {verified}")

    if not verified:
        return []

    # bootstrap a short history for each candidate and measure tick consistency
    scores = {}
    for symbol in verified:
        try:
            resp2 = await client.send({
                "ticks_history": symbol, "count": 200, "end": "latest", "style": "ticks"
            })
            times = resp2.get("history", {}).get("times", [])
            if len(times) < 10:
                continue
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            mean_gap = sum(gaps) / len(gaps)
            std_gap = (sum((g - mean_gap)**2 for g in gaps) / len(gaps)) ** 0.5
            cv = std_gap / mean_gap if mean_gap > 0 else 999
            scores[symbol] = cv
            print(f"[select_top_1hz] {symbol}: mean_gap={mean_gap:.2f}s  cv={cv:.3f}")
        except Exception as e:
            print(f"[select_top_1hz] {symbol}: bootstrap failed: {e}")
        await asyncio.sleep(0.05)

    if not scores:
        print("[select_top_1hz] no consistency data collected, returning all verified (up to n_top)")
        return verified[:n_top]

    ranked = sorted(scores, key=scores.get)          # ascending CV = most consistent first
    top = ranked[:n_top]
    print(f"[select_top_1hz] top {n_top} by tick consistency: {top}")
    return top



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
            "duration": int(duration),   # Deriv requires integer; guard against numpy int / float
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
# ---------------------------------------------------------------------------
# CONFIRMATION LAYERS (L13-L18) — no model fitting needed, evaluate live
# ---------------------------------------------------------------------------
def compute_rsi(prices, period=14):
    """L13a: RSI. Returns (rsi_value, signal) where signal is +1 (oversold,
    expect up), -1 (overbought, expect down), 0 (neutral)."""
    if len(prices) < period + 2:
        return 50.0, 0.0
    deltas = np.diff(prices[-(period + 2):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0, -1.0
    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1 + rs))
    if rsi < 30:
        signal = (30 - rsi) / 30         # +1 at RSI=0, 0 at RSI=30
    elif rsi > 70:
        signal = -(rsi - 70) / 30        # -1 at RSI=100, 0 at RSI=70
    else:
        signal = 0.0
    return float(rsi), float(np.clip(signal, -1, 1))


def compute_stoch_rsi(prices, rsi_period=14, stoch_period=14):
    """L13b: Stochastic RSI. More sensitive than plain RSI for
    short-term overbought/oversold on fast synthetic-index tick data."""
    if len(prices) < rsi_period + stoch_period + 5:
        return 0.5, 0.0
    rsi_series = []
    for i in range(stoch_period):
        idx = -(stoch_period - i + rsi_period)
        rsi_val, _ = compute_rsi(prices[:len(prices) - (stoch_period - i - 1)], rsi_period)
        rsi_series.append(rsi_val)
    rsi_series = np.array(rsi_series)
    lo, hi = np.min(rsi_series), np.max(rsi_series)
    if hi == lo:
        return 0.5, 0.0
    stoch_k = (rsi_series[-1] - lo) / (hi - lo)
    signal = 0.0
    if stoch_k < 0.2:
        signal = (0.2 - stoch_k) / 0.2
    elif stoch_k > 0.8:
        signal = -(stoch_k - 0.8) / 0.2
    return float(stoch_k), float(np.clip(signal, -1, 1))


def compute_adx(prices, period=14):
    """L14: ADX trend-strength filter. ADX > 25 = trending (trust momentum
    layers), ADX < 20 = ranging (trust mean-reversion layers).
    Returns (adx_value, trend_strength_0_to_1, +DI > -DI = up_bias)."""
    if len(prices) < period * 2 + 2:
        return 20.0, 0.3, 0.0
    highs  = prices  # for tick data, use price as both H/L proxy
    lows   = prices
    closes = prices
    n = len(prices)
    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, n):
        tr  = abs(highs[i] - lows[i])
        pdm = max(highs[i] - highs[i-1], 0)
        ndm = max(lows[i-1] - lows[i], 0)
        if pdm < ndm: pdm = 0
        elif ndm < pdm: ndm = 0
        tr_list.append(tr); pdm_list.append(pdm); ndm_list.append(ndm)
    tr_a  = np.array(tr_list[-period*2:])
    pdm_a = np.array(pdm_list[-period*2:])
    ndm_a = np.array(ndm_list[-period*2:])
    atr   = np.mean(tr_a[-period:])
    if atr == 0:
        return 20.0, 0.3, 0.0
    pdi = 100 * np.mean(pdm_a[-period:]) / atr
    ndi = 100 * np.mean(ndm_a[-period:]) / atr
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-9)
    adx = float(np.mean([100 * abs(pdm_a[i] - ndm_a[i]) /
                         (np.mean(tr_a[:i+1]) * period + 1e-9)
                         for i in range(period, len(tr_a))]) if len(tr_a) > period else dx)
    adx = float(np.clip(adx, 0, 100))
    trend_strength = float(np.clip((adx - 20) / 30, 0, 1))  # 0 at ADX=20, 1 at ADX=50
    up_bias = float(np.sign(pdi - ndi))
    return adx, trend_strength, up_bias


def compute_bollinger(prices, period=20, n_std=2.0):
    """L15: Bollinger Band %B. Confirms OU mean-reversion signals.
    %B near 0 = price at lower band (oversold), near 1 = upper band (overbought).
    Signal: positive = expect up (below mid), negative = expect down (above mid)."""
    if len(prices) < period + 2:
        return 0.5, 0.0
    window = prices[-period:]
    mid    = np.mean(window)
    std    = np.std(window)
    if std == 0:
        return 0.5, 0.0
    upper   = mid + n_std * std
    lower   = mid - n_std * std
    pct_b   = (prices[-1] - lower) / (upper - lower + 1e-9)
    pct_b   = float(np.clip(pct_b, -0.5, 1.5))
    signal  = float(np.clip((0.5 - pct_b) * 2, -1, 1))   # +1 at lower band, -1 at upper
    return pct_b, signal


def compute_zscore(prices, period=50):
    """L16: Z-score of current price vs rolling mean. Confirms or contradicts
    OU reversion direction. Strong signal when Z > 2 or < -2."""
    if len(prices) < period + 2:
        return 0.0, 0.0
    window = prices[-period:]
    mu     = np.mean(window)
    sigma  = np.std(window) if np.std(window) > 0 else 1e-9
    z      = (prices[-1] - mu) / sigma
    signal = float(np.clip(-z / 2, -1, 1))   # negative z = below mean = expect up
    return float(z), signal


def transfer_entropy(source_returns, target_returns, lag=1, bins=5):
    """L17: Transfer entropy from source to target. Measures whether source's
    past directional moves provide information about target's next move,
    beyond what target's own history provides. Returns positive value if
    source -> target information flow exists, else near-zero.

    Uses binned estimator for speed (proper KSG estimator is O(n^2)).
    Returns a signed directional signal: positive = source predicts target
    up, negative = source predicts target down."""
    n = min(len(source_returns), len(target_returns)) - lag
    if n < 30:
        return 0.0
    s = source_returns[-n - lag:-lag]
    t_past   = target_returns[-n - lag:-lag]
    t_future = target_returns[-n:]
    try:
        s_bin  = np.digitize(s,       np.percentile(s,       np.linspace(0, 100, bins + 1)[1:-1]))
        tp_bin = np.digitize(t_past,  np.percentile(t_past,  np.linspace(0, 100, bins + 1)[1:-1]))
        tf_bin = np.digitize(t_future,np.percentile(t_future,np.linspace(0, 100, bins + 1)[1:-1]))
        # P(t_future | t_past, s) vs P(t_future | t_past)
        joint3  = np.zeros((bins, bins, bins))
        joint2  = np.zeros((bins, bins))
        joint2b = np.zeros((bins, bins))
        marg    = np.zeros(bins)
        for i in range(n):
            si  = min(s_bin[i],  bins - 1)
            tpi = min(tp_bin[i], bins - 1)
            tfi = min(tf_bin[i], bins - 1)
            joint3[tfi, tpi, si]  += 1
            joint2[tfi, tpi]      += 1
            joint2b[tpi, si]      += 1
            marg[tpi]             += 1
        joint3  = joint3 / (n + 1e-9)
        joint2  = joint2 / (n + 1e-9)
        joint2b = joint2b / (n + 1e-9)
        marg    = marg / (n + 1e-9)
        te = 0.0
        for tfi in range(bins):
            for tpi in range(bins):
                for si in range(bins):
                    num = joint3[tfi, tpi, si]
                    if num <= 0: continue
                    denom_a = joint2b[tpi, si] if joint2b[tpi, si] > 0 else 1e-9
                    denom_b = joint2[tfi, tpi] if joint2[tfi, tpi] > 0 else 1e-9
                    base    = marg[tpi]         if marg[tpi] > 0         else 1e-9
                    te += num * np.log((num * base) / (denom_a * denom_b) + 1e-9)
        # directional component: if source recently moved up, does target follow?
        src_dir = np.sign(np.mean(s[-5:]))
        return float(np.clip(te * src_dir, -1, 1))
    except Exception:
        return 0.0


def detect_jumps(returns, threshold_sigma=2.5):
    """L18: Jump-diffusion — Merton-style jump detection. Identifies ticks
    where the absolute return exceeds threshold_sigma standard deviations
    (likely engineered jumps in synthetic indices). Returns:
      jump_intensity  : recent jump frequency (0-1 normalised)
      jump_direction  : +1 if recent jumps were up, -1 if down, 0 if mixed
      post_jump_signal: after a large jump, expect partial reversion (-jump_dir)"""
    if len(returns) < 30:
        return 0.0, 0.0, 0.0
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0, 0.0, 0.0
    z_scores  = returns / sigma
    jump_mask = np.abs(z_scores) > threshold_sigma
    recent    = jump_mask[-20:]
    intensity = float(np.mean(recent))
    if not np.any(recent):
        return intensity, 0.0, 0.0
    recent_z  = z_scores[-20:]
    jump_dirs = np.sign(recent_z[recent])
    jump_dir  = float(np.mean(jump_dirs)) if len(jump_dirs) > 0 else 0.0
    # post-jump: last tick was a jump → expect partial reversion
    post_jump = -float(np.sign(z_scores[-1])) if jump_mask[-1] else 0.0
    return intensity, float(jump_dir), float(post_jump)


def fit_garch(returns, scale=1000.0):
    if len(returns) < MIN_TICKS_FOR_FIT:
        return None
    try:
        scaled = returns * scale
        am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        # arch's SLSQP optimizer prints convergence diagnostics directly to
        # stdout/stderr on non-convergence, bypassing warnings.filterwarnings.
        # These aren't fatal (a result is still returned) but were showing up
        # as noisy 'error' severity log lines - fully suppress at the source.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = am.fit(disp="off")
        return result
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

    # use the actual measured mean inter-tick gap as dt so OU theta is in
    # real seconds regardless of whether this is a 1HZ (dt~1s) or R_ (dt~2s) symbol
    actual_dt = sd.mean_tick_dt()
    models.tick_dt = actual_dt

    models.hmm_model = fit_hmm(returns)
    models.garch_result = fit_garch(returns, scale=models.garch_scale)
    models.ou_params = fit_ou(prices, dt=actual_dt)
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
    """Evaluates ALL 18 layers using the CACHED fitted models. Returns None if
    no model has been fitted yet (symbol not tradable until first calibration)."""
    if models is None or not models.fitted:
        return None
    returns = sd.returns()
    prices  = sd.prices()
    if len(returns) < MIN_TICKS_LIVE:
        return None

    recent_returns = returns[-50:] if len(returns) >= 50 else returns

    # ── Fitted-model layers (L01-L12) ──────────────────────────────────────
    trend_weight, hmm_lean = hmm_trend_weight(models.hmm_model, recent_returns)
    vol_trust, cond_vol    = garch_vol_trust(models.garch_result, returns, models.garch_scale)
    ou                     = ou_reversion_signal(prices, models.ou_params)

    current_t  = float(sd.epochs()[-1] - models.origin_epoch)
    lam_up     = hawkes_intensity_now(models.hawkes_up,   models.hawkes_up_events,   current_t)
    lam_down   = hawkes_intensity_now(models.hawkes_down, models.hawkes_down_events, current_t)
    hawkes_sig = (lam_up - lam_down) / (lam_up + lam_down + 1e-9)

    h          = hurst_rs(prices)
    arfima     = arfima_bias(returns, h)
    markov_p   = markov_directional_prob(returns)
    kalman     = kalman_trend_filter(prices)
    ent_trust  = sample_entropy_trust(returns[-150:] if len(returns) >= 150 else returns)
    copula     = copula_agreement(sd.symbol, returns_window_dict)

    # ── Confirmation layers (L13-L18) ───────────────────────────────────────
    _,    rsi_signal   = compute_rsi(prices)
    _,    srsi_signal  = compute_stoch_rsi(prices)
    adx_val, adx_trend, adx_dir = compute_adx(prices)
    _,    boll_signal  = compute_bollinger(prices)
    z_val, z_signal    = compute_zscore(prices)

    # transfer entropy: average signal from all OTHER symbols toward this one
    te_signal = 0.0
    others = {s: r for s, r in returns_window_dict.items() if s != sd.symbol}
    if others:
        te_vals = [transfer_entropy(src_r, returns[-200:] if len(returns) >= 200 else returns)
                   for src_r in others.values()]
        te_signal = float(np.mean(te_vals))

    jump_intensity, jump_dir, post_jump = detect_jumps(returns)

    # Hurst-derived arbiter: how much to trust momentum vs reversion layers
    # H > 0.5 → persistent (trust momentum), H < 0.5 → anti-persistent (trust reversion)
    # Expressed as a centred signal so it contributes its own log-odds term
    hurst_signal = float(np.clip((h - 0.5) * 4, -1, 1))   # +1 at H=0.75, -1 at H=0.25

    return {
        # fitted-model layers
        "markov_p":     markov_p,
        "hmm_lean":     hmm_lean,
        "trend_weight": trend_weight,
        "hawkes":       hawkes_sig,
        "ou_dir":       ou["reversion_dir"],
        "ou_strength":  ou["strength"],
        "hurst":        h,
        "hurst_signal": hurst_signal,
        "arfima_bias":  arfima,
        "vol_trust":    vol_trust,
        "entropy_trust":ent_trust,
        "kalman":       kalman,
        "copula_agree": copula,
        "cond_vol":     cond_vol,
        "ou_params":    models.ou_params,
        # confirmation layers
        "rsi_signal":   rsi_signal,
        "srsi_signal":  srsi_signal,
        "adx_val":      adx_val,
        "adx_trend":    adx_trend,
        "adx_dir":      adx_dir,
        "boll_signal":  boll_signal,
        "z_signal":     z_signal,
        "z_val":        z_val,
        "te_signal":    te_signal,
        "jump_intensity": jump_intensity,
        "jump_dir":     jump_dir,
        "post_jump":    post_jump,
        # pass through for calibration weight lookup
        "per_layer_weights": models.per_layer_weights,
    }


def bayesian_fusion(features):
    """Log-odds Bayesian evidence combination across all 18 layers.

    WEIGHT HIERARCHY (highest to lowest precision):
      1. Per-symbol weights learned from OOS correlation during deep calibration
         (stored in features["per_layer_weights"]) — used when available.
      2. Static defaults below — used as fallback for unlearned symbols.

    Hurst contributes its own direct term (not just via ARFIMA) as the
    momentum/reversion arbiter. New confirmation layers (RSI, StochRSI, ADX,
    Bollinger, Z-score, Transfer entropy, Jump-diffusion) add incremental
    evidence without overriding the core fitted-model signal.

    trust_multiplier = vol_trust * entropy_trust gates the entire fusion:
    high-volatility / high-entropy (near-random) conditions suppress all
    evidence proportionally, not just individual layers."""

    learned = features.get("per_layer_weights") or {}

    def W(key, default):
        """Return learned weight if available, else static default."""
        return float(learned.get(key, default))

    p_markov     = float(np.clip(features["markov_p"], 1e-3, 1 - 1e-3))
    markov_logit = math.log(p_markov / (1 - p_markov))
    trend_w      = features["trend_weight"]
    hurst_w      = float(np.clip(features["hurst"], 0, 1))   # H itself as trust scalar

    # ── Core fitted-model evidence ──────────────────────────────────────────
    evidence = [
        # (signal_scaled_to_logit_range, base_weight)
        (markov_logit,                                          W("markov",   1.0)),
        (features["hmm_lean"]    * 2.0,                        W("hmm",      trend_w)),
        (features["hawkes"]      * 2.5,                        W("hawkes",   trend_w)),
        (features["ou_dir"] * features["ou_strength"] * 2.0,  W("ou",       1 - trend_w)),
        (features["hurst_signal"]* 1.2,                        W("hurst",    0.6)),   # ← direct Hurst term
        (features["arfima_bias"] * 1.5,                        W("arfima",   0.55)),
        (features["kalman"]      * 1.5,                        W("kalman",   0.65)),
        ((features["copula_agree"] - 0.5) * 2.0,              W("copula",   0.50)),
    ]

    # ── Confirmation layers (incremental, lower base weight) ────────────────
    adx_trust = features["adx_trend"]   # 0 = ranging, 1 = strongly trending
    # RSI/StochRSI agree on direction: boost weight; disagree: reduce
    rsi_agree = 1.0 if (features["rsi_signal"] * features["srsi_signal"]) >= 0 else 0.4
    # Bollinger and Z-score both measure price stretch — agree: boost
    bz_agree  = 1.0 if (features["boll_signal"] * features["z_signal"]) >= 0 else 0.4

    evidence += [
        (features["rsi_signal"],                               W("rsi",      0.35) * rsi_agree),
        (features["srsi_signal"],                              W("srsi",     0.30) * rsi_agree),
        (features["adx_dir"]     * adx_trust,                 W("adx",      0.35)),
        (features["boll_signal"]                             , W("boll",     0.30) * bz_agree),
        (features["z_signal"],                                 W("zscore",   0.30) * bz_agree),
        (features["te_signal"],                                W("te",       0.30)),
        # Jump: during jump use jump_dir, post-jump use reversion signal
        (features["jump_dir"]    * features["jump_intensity"], W("jump",     0.25)),
        (features["post_jump"]   * features["jump_intensity"], W("post_jump",0.20)),
    ]

    total_trust = features["vol_trust"] * features["entropy_trust"]
    log_odds, total_weight = 0.0, 0.0
    for log_ratio, weight in evidence:
        w = float(weight) * total_trust
        log_odds     += log_ratio * w
        total_weight += abs(w)

    p_up       = float(np.clip(1.0 / (1.0 + math.exp(-log_odds)), 0.01, 0.99))
    confidence = abs(p_up - 0.5) * 2.0 * total_trust
    return p_up, confidence


# ---------------------------------------------------------------------------
# LAYER 12: MONTE CARLO DURATION SELECTOR
# ---------------------------------------------------------------------------
def monte_carlo_duration(prices, returns, direction, feats, candidate_durations, n_sims=MC_SIMULATIONS, models=None):
    """Takes the direction already decided by the Bayesian layer (does NOT
    re-decide direction) and simulates forward paths to find which duration
    maximizes expected win probability.

    OU reversion pull weighted by (1 - trend_weight) - same weighting Bayesian
    fusion used when deciding direction, so MC never silently fights the chosen
    direction (fixed the exp_win=0.00 bug from earlier logs).

    When deep startup calibration has produced empirical per-duration win rates,
    those are blended with the simulation estimate (70% sim / 30% empirical) so
    duration selection is anchored to what actually happened on this symbol."""
    if len(returns) < 20:
        return candidate_durations[0], 0.5

    cond_vol = feats.get("cond_vol")
    vol = cond_vol if cond_vol and cond_vol > 0 else (np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns))
    vol = vol if vol > 0 else 1e-6

    hawkes_signal = feats.get("hawkes", 0.0)
    drift = direction * abs(np.mean(returns[-50:])) * (1 + abs(hawkes_signal) * 0.5) if len(returns) >= 50 else 0.0

    ou_params = feats.get("ou_params")
    trend_weight = feats.get("trend_weight", 0.5)
    current_price = prices[-1]
    reversion_pull = 0.0
    if ou_params and ou_params.get("theta", 0) > 0:
        raw_pull = ou_params["theta"] * (ou_params["mu"] - current_price) * 0.01
        reversion_pull = raw_pull * (1 - trend_weight)

    empirical = getattr(models, "empirical_duration_win_rates", {}) if models else {}

    best = None
    for dur in candidate_durations:
        steps = np.random.normal(drift + reversion_pull, vol, size=(n_sims, dur))
        path_totals = np.sum(steps, axis=1)
        wins = np.sum(path_totals > 0) if direction > 0 else np.sum(path_totals < 0)
        sim_win_rate = wins / n_sims
        blended = 0.70 * sim_win_rate + 0.30 * empirical[dur] if dur in empirical and empirical[dur] > 0 else sim_win_rate
        if best is None or blended > best[1]:
            best = (dur, blended)
    return best


# ---------------------------------------------------------------------------
# ENSEMBLE SELECTOR
# ---------------------------------------------------------------------------
def select_trade(symbol_scores, reliability, global_threshold, per_symbol_threshold=None):
    """Selects the single strongest-signal symbol that clears its own
    per-symbol threshold (derived from that symbol's OOS confidence
    distribution during deep calibration). Falls back to the global threshold
    for symbols without a calibrated per-symbol value.

    Per-symbol thresholds mean a symbol with naturally lower confidence scores
    (e.g. R_10 which is more random) gets judged against its own distribution,
    not penalised against a global bar set by a more predictable symbol."""
    per_sym_thr = per_symbol_threshold or {}
    scored = []
    for symbol, (p_up, confidence) in symbol_scores.items():
        score     = confidence * reliability.get(symbol, 1.0)
        direction = 1 if p_up > 0.5 else -1
        thr       = per_sym_thr.get(symbol, global_threshold)
        scored.append((symbol, direction, p_up, score, thr))

    if not scored:
        return None

    # Filter: each symbol must clear its own threshold
    scored = [s for s in scored if s[3] >= s[4]]
    if not scored:
        return None

    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[0]

    # Gap check: top scorer must lead runner-up meaningfully
    if len(scored) > 1 and (top[3] - scored[1][3]) < MIN_SCORE_GAP:
        return None

    return top[:4]   # (symbol, direction, p_up, score)


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
def explain_signal(symbol, direction, feats, p_up, confidence, duration, exp_win, score):
    """Prints a human-readable breakdown of WHY this trade was taken —
    which layers drove the signal, how strongly, and what the ensemble
    concluded. Logged once at entry before the contract is placed."""
    side     = "CALL (UP)" if direction > 0 else "PUT (DOWN)"
    ts       = datetime.utcnow().isoformat()
    bar      = "█"
    sep      = "─" * 60

    def bar_str(val, width=20):
        """Render a ±1 value as a centred ASCII bar."""
        v     = float(np.clip(val, -1, 1))
        mid   = width // 2
        filled= int(abs(v) * mid)
        if v >= 0:
            return " " * mid + bar * filled + " " * (width - mid - filled)
        else:
            return " " * (mid - filled) + bar * filled + " " * mid + " " * (width - mid)

    # Compile layer contributions into a ranked list
    layer_signals = [
        ("Markov chain",    (feats["markov_p"] - 0.5) * 2),
        ("HMM regime",      feats["hmm_lean"]),
        ("Hawkes momentum", feats["hawkes"]),
        ("OU mean-rev",     feats["ou_dir"] * feats["ou_strength"]),
        ("Hurst",           feats["hurst_signal"]),
        ("ARFIMA long-mem", feats["arfima_bias"]),
        ("Kalman trend",    feats["kalman"]),
        ("Copula agree",    (feats["copula_agree"] - 0.5) * 2),
        ("RSI",             feats["rsi_signal"]),
        ("StochRSI",        feats["srsi_signal"]),
        ("ADX dir",         feats["adx_dir"] * feats["adx_trend"]),
        ("Bollinger %B",    feats["boll_signal"]),
        ("Z-score",         feats["z_signal"]),
        ("Transfer entropy",feats["te_signal"]),
        ("Jump direction",  feats["jump_dir"] * feats["jump_intensity"]),
        ("Post-jump rev",   feats["post_jump"] * feats["jump_intensity"]),
    ]

    # Sort by absolute contribution, strongest first
    layer_signals.sort(key=lambda x: abs(x[1]), reverse=True)

    # Count layers agreeing vs disagreeing with the final direction
    agree    = sum(1 for _, v in layer_signals if v * direction > 0)
    disagree = sum(1 for _, v in layer_signals if v * direction < 0)
    neutral  = len(layer_signals) - agree - disagree

    hurst_regime = ("persistent / trending" if feats["hurst"] > 0.55
                    else "anti-persistent / mean-reverting" if feats["hurst"] < 0.45
                    else "near-random walk")
    hmm_regime   = ("trending"  if feats["trend_weight"] > 0.65
                    else "ranging" if feats["trend_weight"] < 0.4
                    else "mixed")
    vol_state    = ("HIGH — signal down-weighted" if feats["vol_trust"] < 0.5
                    else "ELEVATED" if feats["vol_trust"] < 0.75
                    else "normal")
    entropy_state= ("HIGH — low structure"  if feats["entropy_trust"] < 0.4
                    else "MODERATE" if feats["entropy_trust"] < 0.65
                    else "low — market is structured")

    print(f"\n{sep}")
    print(f"  TRADE SIGNAL  {ts}")
    print(sep)
    print(f"  Symbol  : {symbol}   Direction : {side}")
    print(f"  p(UP)   : {p_up:.4f}   Confidence: {confidence:.4f}   Score: {score:.4f}")
    print(f"  Duration: {duration} ticks   MC exp. win rate: {exp_win:.2%}")
    print(f"  Trust   : vol={feats['vol_trust']:.2f}  entropy={feats['entropy_trust']:.2f}  "
          f"combined={feats['vol_trust']*feats['entropy_trust']:.2f}")
    print("\n  Market regime:")
    print(f"    Hurst H={feats['hurst']:.3f}  → {hurst_regime}")
    print(f"    HMM trend_weight={feats['trend_weight']:.2f}  → {hmm_regime}")
    print(f"    Volatility state → {vol_state}")
    print(f"    Entropy state    → {entropy_state}")
    print(f"\n  Layer breakdown  [{agree} agree | {disagree} disagree | {neutral} neutral]")
    print(f"  {'Layer':<20}  {'Signal':>7}  {'Direction bar (±1)':^22}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*22}")
    for name, val in layer_signals:
        tag = "▲" if val * direction > 0 else ("▼" if val * direction < 0 else "─")
        print(f"  {name:<20}  {val:>+.4f}  {bar_str(val)}  {tag}")
    print(f"\n  Decision: {agree}/{len(layer_signals)} layers support {side}")
    print(sep + "\n")


def log_trade(symbol, direction, stake, won, profit, step):
    ts   = datetime.utcnow().isoformat()
    side = "CALL" if direction > 0 else "PUT"
    print(f"[{ts}] {symbol} {side} step={step} stake={stake:.2f} "
          f"won={won} profit={profit:+.2f}")


def log_trade_summary(symbol, direction, stakes_used, profits, sequence_won,
                      balance_before, balance_after, p_up, confidence, duration):
    """Printed once after a full martingale sequence resolves (win or full loss).
    Gives a compact but complete picture of what happened and what it cost."""
    ts        = datetime.utcnow().isoformat()
    side      = "CALL" if direction > 0 else "PUT"
    n_steps   = len(stakes_used)
    total_staked = sum(stakes_used)
    net_pnl   = sum(profits)
    outcome   = "✓ WON" if sequence_won else "✗ LOST ALL STEPS"
    bal_delta = balance_after - balance_before
    sep       = "─" * 60

    print(f"\n{sep}")
    print(f"  TRADE SUMMARY  {ts}")
    print(sep)
    print(f"  Symbol    : {symbol}   {side}   {duration} ticks")
    print(f"  Signal    : p_up={p_up:.4f}   confidence={confidence:.4f}")
    print(f"  Outcome   : {outcome}")
    print(f"  Steps used: {n_steps} / {MARTINGALE_MAX_STEPS + 1}")
    print(f"  {'Step':<6}  {'Stake':>8}  {'Result':>8}  {'P/L':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i, (s, p) in enumerate(zip(stakes_used, profits)):
        result = "WIN" if p > 0 else "LOSS"
        print(f"  {i:<6}  {s:>8.2f}  {result:>8}  {p:>+8.2f}")
    print(f"  {'TOTAL':<6}  {total_staked:>8.2f}  {'':>8}  {net_pnl:>+8.2f}")
    print(f"\n  Balance : {balance_before:.2f} → {balance_after:.2f}  ({bal_delta:+.2f})")
    print(sep + "\n")


async def execute_single_step(client, state, symbol, direction, stake, step):
    """Places exactly ONE trade and returns. Never loops to the next martingale
    step — that decision belongs to the main signal loop, which waits for a
    genuine quality entry before placing any recovery step."""
    state.trade_in_progress = True
    won, profit = False, 0.0
    try:
        contract_id = await buy_contract(client, symbol, direction, stake, "t", stake)
        won, profit = await wait_for_contract_result(client, contract_id)
        log_trade(symbol, direction, stake, won, profit, step)
    except Exception as e:
        print(f"[Trade] Error on {symbol} step={step}: {e}")

    # accumulate into the sequence tracker for the summary log
    state.seq_stakes.append(stake)
    state.seq_profits.append(profit)

    # step-0 raw signal win-rate tracking (honest edge measurement)
    if step == 0:
        state.step0_total[symbol] += 1
        if won:
            state.step0_wins[symbol] += 1

    try:
        bal_resp = await client.send({"balance": 1})
        state.balance = bal_resp["balance"]["balance"]
    except Exception:
        pass

    state.trade_in_progress = False
    return won, profit


def clear_recovery(state):
    """Reset all recovery context fields — called on sequence win or exhaustion."""
    state.recovery_symbol    = None
    state.recovery_step      = 0
    state.recovery_stake     = 0.0
    state.recovery_direction = 0


def reset_sequence_accumulator(state, balance_now, p_up=0.5, confidence=0.0, duration=0):
    """Called at the START of a new sequence (step=0 entry). Resets all
    per-sequence tracking so the summary log reflects only this sequence."""
    state.seq_stakes         = []
    state.seq_profits        = []
    state.seq_balance_before = balance_now
    state.seq_p_up           = p_up
    state.seq_confidence     = confidence
    state.seq_duration       = duration


def emit_sequence_summary(state, symbol, direction, sequence_won):
    """Called at the END of a sequence. Prints the full trade summary."""
    log_trade_summary(
        symbol        = symbol,
        direction     = direction,
        stakes_used   = list(state.seq_stakes),
        profits       = list(state.seq_profits),
        sequence_won  = sequence_won,
        balance_before= state.seq_balance_before,
        balance_after = state.balance,
        p_up          = state.seq_p_up,
        confidence    = state.seq_confidence,
        duration      = state.seq_duration,
    )


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
    (hit_rate, fitted_models, confidences) - the same models get cached for
    live trading if validation passes a sane bar, and `confidences` (the raw
    confidence score at each replayed point) feeds the adaptive threshold
    calibration in run_calibration."""
    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT + 100:
        return 0.5, None, []

    split = max(MIN_TICKS_FOR_FIT, int(n_ticks * train_frac))
    train_sd = sd.slice_copy(split)
    models = fit_symbol_models(train_sd)
    if not models.fitted:
        return 0.5, None, []

    eval_sd = sd.slice_copy(split)
    remaining_ticks = list(sd.ticks)[split:]
    hits, total = 0, 0
    confidences = []
    for i in range(0, len(remaining_ticks) - horizon, step):
        eval_sd.add_tick(*remaining_ticks[i])
        feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
        if feats is None:
            continue
        p_up, confidence = bayesian_fusion(feats)
        confidences.append(confidence)
        predicted_dir = 1 if p_up > 0.5 else -1
        current_price = remaining_ticks[i][1]
        future_price = remaining_ticks[i + horizon][1]
        actual_dir = 1 if future_price > current_price else -1
        hits += int(predicted_dir == actual_dir)
        total += 1

    hit_rate = hits / total if total > 0 else 0.5
    return hit_rate, models, confidences



# ---------------------------------------------------------------------------
# DEEP STARTUP CALIBRATION
# ---------------------------------------------------------------------------
def expanding_window_walk_forward(sd, n_folds=5, horizons=None, step=3):
    """True expanding-window walk-forward: models are REFITTED at each fold
    boundary on all data up to that point, then evaluated on the next unseen
    window. Returns a full report including per-fold hit rates, per-duration
    empirical win rates, per-layer correlations, and models fitted on the
    complete dataset for live trading."""
    if horizons is None:
        horizons = CANDIDATE_DURATIONS

    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT * 2 + 100:
        return None

    all_ticks = list(sd.ticks)
    fold_size = (n_ticks - MIN_TICKS_FOR_FIT) // (n_folds + 1)
    if fold_size < 30:
        return None

    per_fold_hit_rates = []
    per_duration_outcomes = defaultdict(lambda: [0, 0])
    layer_outcomes = defaultdict(list)
    all_confidences = []
    mid_h = horizons[len(horizons) // 2]

    for fold in range(n_folds):
        train_end = MIN_TICKS_FOR_FIT + fold_size * (fold + 1)
        test_end  = min(train_end + fold_size, n_ticks)
        if test_end - train_end < 20:
            continue

        train_sd = sd.slice_copy(train_end)
        models   = fit_symbol_models(train_sd)
        if not models.fitted:
            continue

        eval_sd    = sd.slice_copy(train_end)
        test_ticks = all_ticks[train_end:test_end]
        hits_fold, total_fold = 0, 0

        for i in range(0, len(test_ticks) - max(horizons), step):
            eval_sd.add_tick(*test_ticks[i])
            feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
            if feats is None:
                continue
            p_up, confidence = bayesian_fusion(feats)
            all_confidences.append(confidence)
            predicted_dir = 1 if p_up > 0.5 else -1
            current_price = test_ticks[i][1]

            for h in horizons:
                if i + h >= len(test_ticks):
                    continue
                future_price = test_ticks[i + h][1]
                actual_dir   = 1 if future_price > current_price else -1
                won = int(predicted_dir == actual_dir)
                per_duration_outcomes[h][0] += won
                per_duration_outcomes[h][1] += 1
                if h == mid_h:
                    hits_fold  += won
                    total_fold += 1

            # per-layer correlation data (mid horizon only) — all 18 layers
            if i + mid_h < len(test_ticks):
                actual_mid = 1 if test_ticks[i + mid_h][1] > current_price else -1
                for layer, key in [
                    ("markov",    "markov_p"),    ("hmm",       "hmm_lean"),
                    ("hawkes",    "hawkes"),       ("ou",        "ou_dir"),
                    ("hurst",     "hurst_signal"), ("arfima",    "arfima_bias"),
                    ("kalman",    "kalman"),       ("copula",    "copula_agree"),
                    ("vol_trust", "vol_trust"),    ("entropy",   "entropy_trust"),
                    ("rsi",       "rsi_signal"),   ("srsi",      "srsi_signal"),
                    ("adx",       "adx_dir"),      ("boll",      "boll_signal"),
                    ("zscore",    "z_signal"),     ("te",        "te_signal"),
                    ("jump",      "jump_dir"),     ("post_jump", "post_jump"),
                ]:
                    val = feats.get(key)
                    if val is not None:
                        layer_outcomes[layer].append((float(val), actual_mid))

        if total_fold > 0:
            per_fold_hit_rates.append((fold, train_end, total_fold, hits_fold / total_fold))

    if not per_fold_hit_rates:
        return None

    fold_hrs = [x[3] for x in per_fold_hit_rates]
    per_duration_win_rates = {
        dur: wins / total if total > 0 else 0.5
        for dur, (wins, total) in per_duration_outcomes.items()
    }
    per_layer_correlations = {}
    for layer, pairs in layer_outcomes.items():
        if len(pairs) < 20:
            continue
        vals     = np.array([p[0] for p in pairs])
        outcomes = np.array([1 if p[1] > 0 else 0 for p in pairs])
        if np.std(vals) > 0:
            per_layer_correlations[layer] = float(np.corrcoef(vals, outcomes)[0, 1])

    best_models = fit_symbol_models(sd)

    return {
        "per_fold_hit_rates":      per_fold_hit_rates,
        "per_duration_win_rates":  per_duration_win_rates,
        "per_layer_correlations":  per_layer_correlations,
        "mean_hit_rate":           float(np.mean(fold_hrs)),
        "std_hit_rate":            float(np.std(fold_hrs)),
        "all_confidences":         all_confidences,
        "best_models":             best_models,
        "is_tradeable":            float(np.mean(fold_hrs)) >= 0.46 and best_models.fitted,
        "n_folds_completed":       len(per_fold_hit_rates),
    }


def check_model_stability(models, symbol):
    """Audit fitted model parameters for physical sanity. Returns a list of
    warning strings (empty = clean)."""
    warns = []
    if models.garch_result is not None:
        try:
            p = models.garch_result.params
            alpha = p.get("alpha[1]", p.get("alpha", None))
            beta  = p.get("beta[1]",  p.get("beta",  None))
            if alpha is not None and beta is not None:
                persistence = float(alpha) + float(beta)
                if persistence >= 1.0:
                    warns.append(f"GARCH persistence={persistence:.3f} >= 1.0 (non-stationary)")
                elif persistence > 0.98:
                    warns.append(f"GARCH persistence={persistence:.3f} near-unit-root")
        except Exception:
            pass
    for label, h in [("up", models.hawkes_up), ("down", models.hawkes_down)]:
        if h is not None:
            alpha, beta = h.get("alpha", 0), h.get("beta", 1)
            ratio = alpha / beta if beta > 0 else 999
            if ratio >= 1.0:
                warns.append(f"Hawkes {label}: branching ratio={ratio:.3f} >= 1.0 (explosive)")
            elif ratio > 0.9:
                warns.append(f"Hawkes {label}: branching ratio={ratio:.3f} near-critical")
    if models.ou_params is not None:
        theta = models.ou_params.get("theta", 0)
        if theta <= 0:
            warns.append(f"OU theta={theta:.4f} <= 0 (divergent)")
    if models.hmm_model is not None:
        try:
            for i, p in enumerate(models.hmm_model.get_stationary_distribution()):
                if p < 0.05:
                    warns.append(f"HMM state {i} stationary prob={p:.3f} (degenerate)")
        except Exception:
            pass
    return warns


async def deep_startup_calibration(state, symbol_data, symbols):
    """Full-power startup calibration. Every symbol, every layer, no shortcuts.
    Called ONCE before the bot places any trade. Periodic run_calibration()
    continues every 2 hours and on loss triggers - those are lighter (top-K).
    This is the one time with no time pressure, so we use it fully."""
    state.trading_locked = True
    start = time.time()
    print("=" * 60)
    print("DEEP STARTUP CALIBRATION — full power, all symbols")
    print("=" * 60)

    all_confidences = []
    symbol_reports  = {}

    for s in symbols:
        sd = symbol_data[s]
        n  = len(sd.ticks)
        fam = "1HZ" if "1HZ" in s else "R_ "
        print(f"\n[DeepCal] [{fam}] {s}: {n} ticks  tick_dt={sd.tick_dt:.1f}s — "
              f"starting {5}-fold expanding walk-forward...")

        if n < MIN_TICKS_FOR_FIT * 2 + 100:
            print(f"[DeepCal] {s}: insufficient history, skipping.")
            state.reliability[s] = 0.3
            continue

        report = expanding_window_walk_forward(sd, n_folds=5,
                                               horizons=CANDIDATE_DURATIONS, step=3)
        if report is None:
            print(f"[DeepCal] {s}: walk-forward returned no result. Not tradeable.")
            state.reliability[s] = 0.3
            continue

        stability_warns = check_model_stability(report["best_models"], s)

        print(f"[DeepCal] {s}: {report['n_folds_completed']}/5 folds")
        print(f"  Mean OOS hit rate : {report['mean_hit_rate']:.3f}  (std={report['std_hit_rate']:.3f})")
        print(f"  Per-fold          : {[f'f{x[0]}={x[3]:.3f}' for x in report['per_fold_hit_rates']]}")
        print(f"  Per-duration win% : { {d: f'{v:.3f}' for d,v in sorted(report['per_duration_win_rates'].items())} }")
        print(f"  Layer correlations: { {l: f'{v:+.3f}' for l,v in sorted(report['per_layer_correlations'].items(), key=lambda x: abs(x[1]), reverse=True)} }")
        print(f"  Is tradeable      : {report['is_tradeable']}  (mean hit rate >= 0.46)")
        if stability_warns:
            print(f"  *** STABILITY WARNINGS ***")
            for w in stability_warns:
                print(f"      {w}")
        else:
            print(f"  Model stability   : CLEAN")

        if report["best_models"] is not None and report["best_models"].fitted:
            m = report["best_models"]
            m.empirical_duration_win_rates = report["per_duration_win_rates"]

            # ── Convert OOS per-layer correlations → fusion weights ────────
            # Correlation with realized outcome tells us how much each layer
            # actually predicts direction on THIS specific symbol. We scale
            # it into a positive weight: perfectly correlated layer gets 2x
            # its static default, uncorrelated gets 0.1x (not zero — avoids
            # a layer being silenced on a short OOS window that may be noisy).
            corr = report["per_layer_correlations"]
            if corr:
                learned_w = {}
                for layer, c in corr.items():
                    # abs(corr) in [0,1] → weight in [0.1, 2.0]
                    learned_w[layer] = float(np.clip(0.1 + abs(c) * 1.9, 0.1, 2.0))
                    # preserve sign: if layer is negatively correlated, flip
                    # its evidence contribution (handled in bayesian_fusion via
                    # the weight staying positive but the signal itself carrying
                    # direction - weight scales magnitude only)
                m.per_layer_weights = learned_w
                top3 = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  Learned weights   : top-3 predictors = "
                      f"{[(l, f'{c:+.3f}') for l,c in top3]}")
            else:
                m.per_layer_weights = None
                print(f"  Learned weights   : insufficient OOS data, using static defaults")

            state.model_cache[s] = m

        state.reliability[s] = float(np.clip(report["mean_hit_rate"] / 0.5, 0.3, 1.5))
        symbol_reports[s]    = report

        # ── Per-symbol threshold from THIS symbol's OOS confidence distribution
        # This is the key fix: each symbol gets its own threshold derived from
        # its own OOS confidence scores, not a pooled global number.
        sym_confidences = report["all_confidences"]
        if sym_confidences:
            sym_thr = float(np.clip(
                np.percentile(sym_confidences, ADAPTIVE_THRESHOLD_PERCENTILE), 0.02, 0.55))
            # Quality gate: if fewer than 15% of OOS points clear this bar,
            # lower it slightly to avoid starving good trades. If > 60% clear,
            # raise it slightly to tighten precision.
            pct_clr = float(np.mean(np.array(sym_confidences) >= sym_thr))
            if pct_clr < 0.15:
                sym_thr = float(np.percentile(sym_confidences, max(ADAPTIVE_THRESHOLD_PERCENTILE - 15, 40)))
            elif pct_clr > 0.60:
                sym_thr = float(np.percentile(sym_confidences, min(ADAPTIVE_THRESHOLD_PERCENTILE + 10, 80)))
            state.per_symbol_threshold[s] = sym_thr
            print(f"  Per-symbol thr    : {sym_thr:.4f}  ({pct_clr*100:.0f}% OOS points clear)")
        else:
            state.per_symbol_threshold[s] = state.adaptive_threshold

        all_confidences.extend(sym_confidences)
        print(f"  Reliability       : {state.reliability[s]:.3f}")

    if all_confidences:
        global_thr = float(np.clip(
            np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE), 0.03, 0.6))
        state.adaptive_threshold = global_thr   # global fallback only
        print(f"\n[DeepCal] Global fallback threshold -> {global_thr:.4f} "
              f"(per-symbol thresholds take precedence when set)")
    else:
        print(f"\n[DeepCal] WARNING: no confidence samples — keeping default "
              f"threshold={state.adaptive_threshold:.3f}")

    tradeable     = [s for s,r in symbol_reports.items() if r["is_tradeable"]]
    not_tradeable = [s for s,r in symbol_reports.items() if not r["is_tradeable"]]
    print(f"\n[DeepCal] TRADEABLE ({len(tradeable)}): {tradeable}")
    print(f"[DeepCal] BELOW EDGE BAR ({len(not_tradeable)}): {not_tradeable}")
    print(f"[DeepCal] Below-bar symbols still compete via ensemble — "
          f"lower reliability multiplier means they need a stronger signal to win selection.")

    elapsed = time.time() - start
    print(f"\n[DeepCal] Complete in {elapsed:.1f}s ({elapsed/60:.1f} min). Bot armed.")
    print("=" * 60)

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end       = time.time()
    state.last_activity              = time.time()
    state.trading_locked             = False



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

    all_confidences = []
    for s in candidates:
        sd = symbol_data[s]
        if len(sd.ticks) < MIN_TICKS_FOR_FIT + 100:
            print(f"[Calibrator] {s}: not enough ticks yet, skipping this cycle.")
            continue
        hit_rate, models, confidences = walk_forward_validate(sd)
        if models is not None:
            state.model_cache[s] = models
        state.reliability[s] = float(np.clip(hit_rate / 0.5, 0.3, 1.5))
        state.consecutive_losses[s] = 0
        all_confidences.extend(confidences)
        print(f"[Calibrator] {s}: walk-forward hit_rate={hit_rate:.3f} reliability={state.reliability[s]:.2f} "
              f"n_confidence_samples={len(confidences)}")

    if all_confidences:
        new_threshold = float(np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE))
        # never let the bar collapse to ~0 (untradeable noise floor) or demand
        # near-impossible confidence - keep it in a sane band regardless of
        # what the percentile math produces on a weird sample
        new_threshold = float(np.clip(new_threshold, 0.03, 0.6))
        old_threshold = state.adaptive_threshold
        state.adaptive_threshold = new_threshold
        pct_clearing = float(np.mean(np.array(all_confidences) >= new_threshold)) * 100
        print(f"[Calibrator] adaptive_threshold {old_threshold:.3f} -> {new_threshold:.3f} "
              f"(P{ADAPTIVE_THRESHOLD_PERCENTILE} of {len(all_confidences)} samples, "
              f"~{pct_clearing:.0f}% of replayed points would clear it)")
    else:
        print(f"[Calibrator] no confidence samples collected this cycle - "
              f"keeping threshold at {state.adaptive_threshold:.3f}")

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end = time.time()
    state.last_activity = time.time()
    print(f"[Calibrator] complete in {state.last_calibration_end - start:.1f}s. Updated: {candidates}")
    state.trading_locked = False


# ---------------------------------------------------------------------------
# STREAM CONSUMERS
# ---------------------------------------------------------------------------
async def tick_consumer(queue, symbol_data, state):
    while True:
        data = await queue.get()
        tick = data.get("tick")
        if not tick:
            continue
        symbol = tick.get("symbol")
        if symbol in symbol_data:
            symbol_data[symbol].add_tick(tick["epoch"], tick["quote"])
        state.last_activity = time.time()


async def balance_consumer(queue, state):
    while True:
        data = await queue.get()
        bal = data.get("balance")
        if bal:
            state.balance = bal["balance"]


async def watchdog(state):
    """If WATCHDOG_TIMEOUT seconds pass with no tick received and no main-loop
    iteration completed (state.last_activity untouched), the process is
    assumed locked up. Rather than depending on any specific host's restart
    policy, this re-execs the current Python process in place - identical
    behavior on Railway and on a local PC, no external supervisor needed."""
    while True:
        await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s (limit {WATCHDOG_TIMEOUT}s). "
                  f"Restarting process in place now.")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)


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

    # --- R_ symbols ---
    r_symbols = []
    for attempt in range(1, 6):
        r_symbols = await fetch_tradable_symbols(client)
        if r_symbols:
            break
        print(f"[main] No R_ symbols on attempt {attempt}/5, retrying in 3s...")
        await asyncio.sleep(3)
    if not r_symbols:
        raise RuntimeError("No R_ rise/fall symbols found (check API credentials/connectivity).")

    # --- top-3 1HZ symbols by tick consistency ---
    hz_symbols = []
    for attempt in range(1, 4):
        hz_symbols = await select_top_1hz(client, n_top=3)
        if hz_symbols:
            break
        print(f"[main] No 1HZ symbols on attempt {attempt}/3, retrying in 3s...")
        await asyncio.sleep(3)
    if not hz_symbols:
        print("[main] WARNING: no 1HZ symbols available - proceeding with R_ only.")

    symbols = r_symbols + hz_symbols
    print(f"\nFull tradable universe ({len(symbols)} symbols):")
    print(f"  R_ ({len(r_symbols)}): {r_symbols}")
    print(f"  1HZ top-3 ({len(hz_symbols)}): {hz_symbols}")

    # build SymbolData with correct tick_dt per family
    symbol_data = {}
    for s in r_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=2.0)   # R_ tick ~every 2s
    for s in hz_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=1.0)   # 1HZ ticks every 1s

    print("Bootstrapping tick history for all symbols (this funds the deep startup calibration)...")
    for s in symbols:
        history = await fetch_history(client, s)
        for epoch, price in history:
            symbol_data[s].add_tick(epoch, price)
        actual_dt = symbol_data[s].mean_tick_dt()
        print(f"  {s}: {len(symbol_data[s].ticks)} ticks loaded  actual_mean_dt={actual_dt:.2f}s")

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

    asyncio.create_task(tick_consumer(tick_queue, symbol_data, state))
    asyncio.create_task(balance_consumer(balance_queue, state))
    asyncio.create_task(watchdog(state))

    print("Running initial full-power calibration across the entire universe before trading begins...")
    await deep_startup_calibration(state, symbol_data, symbols)

    print("Bot running. Entering main decision loop.")
    last_heartbeat = 0.0

    while True:
        await asyncio.sleep(2)
        state.last_activity = time.time()

        if state.trading_locked or state.trade_in_progress:
            continue

        trigger = check_calibration_triggers(state)
        if trigger:
            await run_calibration(state, symbol_data, symbols, trigger)
            continue

        ready_symbols = [s for s in symbols
                         if s in state.model_cache
                         and len(symbol_data[s].ticks) >= MIN_TICKS_LIVE]

        now = time.time()
        if now - last_heartbeat > 30:
            rec = (f" | RECOVERY {state.recovery_symbol} "
                   f"step={state.recovery_step} stake={state.recovery_stake:.2f}"
                   if state.recovery_symbol else "")
            s0_parts = []
            for sym in ready_symbols:
                tot = state.step0_total[sym]
                if tot > 0:
                    wr = state.step0_wins[sym] / tot
                    s0_parts.append(f"{sym}:{wr:.0%}({tot})")
            s0_str = " s0_wr=[" + " ".join(s0_parts) + "]" if s0_parts else ""
            print(f"[scan] balance={state.balance:.2f} | "
                  f"{len(ready_symbols)}/{len(symbols)} ready{rec}{s0_str}")
            last_heartbeat = now

        if not ready_symbols:
            continue

        returns_window_dict = {s: symbol_data[s].returns()[-200:] for s in ready_symbols}

        # ── RECOVERY MODE ────────────────────────────────────────────────────
        # A previous step lost. Symbol is locked. We still demand a full-quality
        # signal before placing the recovery step — stake is larger, bar is identical.
        if state.recovery_symbol:
            rec_sym = state.recovery_symbol
            if rec_sym not in ready_symbols:
                print(f"[Recovery] {rec_sym} no longer ready — abandoning sequence.")
                clear_recovery(state)
                state.consecutive_losses[rec_sym] += 1
                continue

            sd = symbol_data[rec_sym]
            feats = compute_features(sd, state.model_cache.get(rec_sym), returns_window_dict)
            if feats is None:
                continue

            p_up, confidence = bayesian_fusion(feats)
            score     = confidence * state.reliability.get(rec_sym, 1.0)
            sym_thr   = state.per_symbol_threshold.get(rec_sym, state.adaptive_threshold)
            signal_dir = 1 if p_up > 0.5 else -1

            if score < sym_thr:
                continue   # signal not strong enough yet — keep waiting

            # Direction-flip guard: if the signal has reversed against the
            # original trade direction, wait for realignment rather than
            # recovering into a confirmed headwind
            if signal_dir != state.recovery_direction:
                continue

            duration, exp_win_rate = monte_carlo_duration(
                sd.prices(), sd.returns(), state.recovery_direction,
                feats, CANDIDATE_DURATIONS,
                models=state.model_cache.get(rec_sym)
            )
            if exp_win_rate < MIN_EXP_WIN_RATE:
                continue

            explain_signal(
                symbol=rec_sym, direction=state.recovery_direction,
                feats=feats, p_up=p_up, confidence=confidence,
                duration=duration, exp_win=exp_win_rate, score=score
            )

            won, _ = await execute_single_step(
                client, state, rec_sym,
                state.recovery_direction, state.recovery_stake, state.recovery_step
            )

            if won:
                print(f"[Recovery] {rec_sym} recovered at step={state.recovery_step}.")
                state.consecutive_losses[rec_sym] = 0
                emit_sequence_summary(state, rec_sym, state.recovery_direction, True)
                clear_recovery(state)
            else:
                next_step  = state.recovery_step + 1
                next_stake = round(state.recovery_stake * MARTINGALE_FACTOR, 2)
                if next_step > MARTINGALE_MAX_STEPS:
                    print(f"[Recovery] {rec_sym} exhausted all {MARTINGALE_MAX_STEPS} steps — sequence closed.")
                    state.consecutive_losses[rec_sym] += 1
                    emit_sequence_summary(state, rec_sym, state.recovery_direction, False)
                    clear_recovery(state)
                    # ── Post-loss deep recal: re-fit ALL symbols, then resume
                    # open scanning with no symbol/direction lock. ──────────
                    print(f"[Recovery] Triggering post-loss deep recalibration across all symbols.")
                    await deep_startup_calibration(state, symbol_data, symbols)
                else:
                    state.recovery_step  = next_step
                    state.recovery_stake = next_stake
                    print(f"[Recovery] {rec_sym} step lost — "
                          f"waiting for signal before step={next_step} stake={next_stake:.2f}")

            state.last_activity = time.time()
            continue

        # ── NORMAL ENTRY ─────────────────────────────────────────────────────
        symbol_scores = {}
        for s in ready_symbols:
            sd    = symbol_data[s]
            feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
            if feats is None:
                continue
            p_up, confidence = bayesian_fusion(feats)
            symbol_scores[s] = (p_up, confidence)

        pick = select_trade(
            symbol_scores, state.reliability,
            state.adaptive_threshold,
            state.per_symbol_threshold
        )
        if not pick:
            continue

        symbol, direction, p_up, score = pick
        sd    = symbol_data[symbol]
        feats = compute_features(sd, state.model_cache.get(symbol), returns_window_dict)

        duration, exp_win_rate = monte_carlo_duration(
            sd.prices(), sd.returns(), direction, feats, CANDIDATE_DURATIONS,
            models=state.model_cache.get(symbol)
        )
        if exp_win_rate < MIN_EXP_WIN_RATE:
            print(f"Skipping {symbol} — MC best win rate {exp_win_rate:.2f} below floor.")
            continue

        base_stake = calculate_stake(state.balance)
        all_stakes = martingale_stakes(base_stake)

        # reset sequence accumulator before step=0 fires
        reset_sequence_accumulator(state, state.balance, p_up, confidence, duration)

        explain_signal(
            symbol=symbol, direction=direction,
            feats=feats, p_up=p_up, confidence=confidence,
            duration=duration, exp_win=exp_win_rate, score=score
        )

        won, _ = await execute_single_step(
            client, state, symbol, direction, base_stake, 0
        )

        if won:
            state.consecutive_losses[symbol] = 0
            emit_sequence_summary(state, symbol, direction, True)
        else:
            if MARTINGALE_MAX_STEPS >= 1:
                state.recovery_symbol    = symbol
                state.recovery_direction = direction
                state.recovery_step      = 1
                state.recovery_stake     = all_stakes[1]
                print(f"[Recovery] step=0 lost on {symbol} — "
                      f"waiting for signal before step=1 stake={all_stakes[1]:.2f}")
            else:
                state.consecutive_losses[symbol] += 1
                emit_sequence_summary(state, symbol, direction, False)
                # martingale disabled — still recalibrate after a loss so the
                # next entry uses fresh models across all symbols/directions
                print(f"[Recovery] Single-step loss on {symbol} — triggering deep recalibration.")
                await deep_startup_calibration(state, symbol_data, symbols)

        state.last_activity = time.time()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[main] Unhandled exception, restarting process in place: {type(e).__name__}: {e}")
        sys.stdout.flush()
        time.sleep(3)  # brief pause so a fast crash loop doesn't hammer the API
        os.execv(sys.executable, [sys.executable] + sys.argv)
