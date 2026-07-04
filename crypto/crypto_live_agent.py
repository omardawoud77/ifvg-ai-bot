"""
Crypto Live Trading Agent — Multi-Symbol, Binance Futures Testnet
==================================================================
Runs the trained MTF agent live on Binance USDT-M perpetual futures testnet
across multiple symbols simultaneously. Each symbol has its own state,
memory, regret review, and dynamic SL/TP — but shares one PPO model and
one global kill switch.

Usage:
  export BINANCE_API_KEY=your_key
  export BINANCE_SECRET_KEY=your_secret
  python3 crypto_live_agent.py

Both LONG and SHORT supported. 3× leverage.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import shutil
import tempfile
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from stable_baselines3 import PPO
from crypto_env_v2 import CryptoMTFEnv, CryptoTechEnv  # NEW_MODEL: import 44-feature env
from reasoning_engine import perceive, interpret, decide, get_dynamic_sl_tp, simulate_missed_trade
from trade_memory import TradeMemory
from sentiment_agent  import SentimentAgent
from position_auditor import PositionAuditor

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOLS         = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRADE_NOTIONAL  = 7000.0         # USDT notional cap per trade per symbol
LEVERAGE        = 5              # 5× leverage
FETCH_EVERY     = 60             # seconds between cycles (1 min)
MAX_DAILY_TRADES = 3             # max NEW entries per UTC day PER SYMBOL
MAX_CONSECUTIVE_LOSSES = 3       # disable symbol after N consecutive losses
MAX_CONCURRENT_POSITIONS = 2     # max symbols open simultaneously (correlation risk)
RISK_PER_TRADE_BY_TIER = {
    'A_PLUS': 0.020,  # 2.0% — high conviction → ~$100 risk on $5000
    'A':      0.020,  # 2.0% — high conviction → ~$100 risk on $5000
    'B':      0.010,  # 1.0% — medium conviction → ~$50 risk
    'C':      0.000,    # DISABLED: live data shows -$130 on 7 WEAK_EXECUTE trades vs +$177 on 4 EXECUTE trades
    'TRASH':  0.000,  # don't trade
}
KILL_SWITCH_DRAWDOWN_PCT = 0.05  # SAFETY FIX: 5% drawdown from starting balance kills all trading
MAX_HOLD_HOURS  = 48             # HORIZON_FIX: match training max_hold_bars=48 (1H bars).
                                 # Was 8h — the model learned exits on a 48h horizon,
                                 # force-closing at 8h invalidated its learned policy.
MODEL_PATH      = "crypto_mtf_balanced_best.zip"  # BALANCED: bear-balanced model, 63.6% WR, +0.60R expectancy
LOG_PATH        = "crypto_live_agent.log"
LEGACY_BTC_STATE = "crypto_live_state.json"  # pre-multisymbol path, migrated on first load
MIN_BARS        = 50

FUTURES_TESTNET_URL = "https://testnet.binancefuture.com/fapi"

HERE = os.path.dirname(os.path.abspath(__file__))

# Persistent dir (Railway volume via PERSIST_DATA_DIR) for everything the bot
# must remember across redeploys: position state, trade memory, logs, disabled
# symbols. Falls back to HERE when the env var is unset (local runs).
PERSIST_DIR = os.environ.get("PERSIST_DATA_DIR", HERE)

def _persist_path(filename):
    """Path on the persistent volume; seeded once from the repo copy if the
    volume doesn't have the file yet (e.g. calibrated trade_memory json)."""
    if PERSIST_DIR == HERE:
        return os.path.join(HERE, filename)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    dst = os.path.join(PERSIST_DIR, filename)
    src = os.path.join(HERE, filename)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy2(src, dst)
    return dst

DISABLED_SYMBOLS_PATH = _persist_path("disabled_symbols.json")  # SAFETY FIX: persist across restarts

def state_path_for(symbol):
    return _persist_path(f"state_{symbol.lower()}.json")

def memory_path_for(symbol):
    """BTC keeps the legacy filename for backward compat with the calibrated regret memory."""
    if symbol == "BTCUSDT":
        return _persist_path("trade_memory.json")
    return _persist_path(f"trade_memory_{symbol.lower()}.json")

def trade_log_path_for(symbol):
    if symbol == "BTCUSDT":
        return _persist_path("trade_log.csv")
    return _persist_path(f"trade_log_{symbol.lower()}.csv")

def trade_events_path_for(symbol):
    return _persist_path(f"trade_events_{symbol.lower()}.jsonl")

def log_trade_event(symbol, event_type, action, qty, price, trade_id="",
                    sl_price=0.0, tp_pct=0.0, pnl_usdt=0.0, pnl_pct=0.0,
                    close_reason="", confidence=0.0, verdict=""):
    """Append a structured JSON line to trade_events_{symbol}.jsonl."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_id,
        "symbol": symbol,
        "event_type": event_type,
        "action": action,
        "qty": qty,
        "price": price,
        "sl_price": sl_price,
        "tp_pct": tp_pct,
        "pnl_usdt": pnl_usdt,
        "pnl_pct": pnl_pct,
        "close_reason": close_reason,
        "confidence": confidence,
        "verdict": verdict,
    }
    path = trade_events_path_for(symbol)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        log.error(f"[{symbol[:3]}] Failed to write trade event: {e}")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "position": 0,
    "entry_price": 0.0,
    "entry_time": None,
    "qty": 0.0,
    "trade_count": 0,
    "wins": 0,
    "losses": 0,
    "total_pnl_usdt": 0.0,
    "daily_trades": 0,
    "last_trade_date": None,
    "sl_pct": 0.015,
    "sl_price": 0.0,  # SL_SOFTWARE: absolute SL price for software-managed stop
    "tp_pct": 0.03,
    "entry_conditions": None,
    "entry_confidence": 0.5,
    "entry_verdict": None,
    "entry_reasoning": "",
    "last_rejected_action": None,
    "last_rejected_conditions": None,
    "last_rejected_bar_ts": None,
    "last_rejected_sl_pct": 0.015,
    "last_rejected_tp_pct": 0.04,
    "breakeven_set": False,
    "trail_1r_set": False,
    "consecutive_losses": 0,
    "last_trade_id": None,
    "pending_signal": None,        # 1H-approved signal awaiting 15m confirmation
    "last_15m_check": None,        # ISO timestamp of last 15m confirmation check
}

def load_state(state_path):
    # SAFETY FIX: wrap json.load in try/except — never crash on corrupted state
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                loaded = json.load(f)
            for k, v in DEFAULT_STATE.items():
                loaded.setdefault(k, v)
            return loaded
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            log.warning(f"⚠️  Corrupted state file {state_path}: {e} — using defaults")
            return dict(DEFAULT_STATE)
    return dict(DEFAULT_STATE)

def save_state(state, state_path):
    """Atomic write: write to tmp file in the same directory, then os.replace().
    Prevents JSON corruption if the process crashes mid-write."""
    dir_name = os.path.dirname(state_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
    except Exception:
        # Clean up the temp file if replace failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def load_state_with_migration(symbol):
    """For BTCUSDT, migrate from the legacy single-symbol state file if needed."""
    new_path = state_path_for(symbol)
    if os.path.exists(new_path):
        return load_state(new_path)

    if symbol == "BTCUSDT":
        legacy = os.path.join(HERE, LEGACY_BTC_STATE)
        if os.path.exists(legacy):
            log.info(f"📦 Migrating BTC state from {LEGACY_BTC_STATE} → state_btcusdt.json")
            # SAFETY FIX: wrap legacy json.load in try/except
            try:
                with open(legacy) as f:
                    loaded = json.load(f)
                for k, v in DEFAULT_STATE.items():
                    loaded.setdefault(k, v)
                save_state(loaded, new_path)
                return loaded
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                log.warning(f"⚠️  Corrupted legacy state {legacy}: {e} — using defaults")
                return dict(DEFAULT_STATE)

    return dict(DEFAULT_STATE)

# SAFETY FIX: return None on failure — never return a fake balance
def fetch_live_balance(client):
    try:
        bal = client.futures_account_balance()
        usdt = next((float(x['balance']) for x in bal if x['asset'] == 'USDT'), 0.0)
        return usdt
    except Exception as e:
        log.error(f"❌ Could not fetch balance: {e}")
        return None  # SAFETY FIX: callers must handle None explicitly


# ── RATE_LIMIT: global Binance IP-ban guard ───────────────────────────────────
# Binance -1003 bans are per-IP and EXTEND on every request made while banned.
# When we see one, parse the 'banned until <ms>' timestamp and go fully silent
# on the data API until it passes (+buffer). Retrying during a ban makes the
# ban longer — the only correct response is zero requests.

_BAN_UNTIL_TS = 0.0        # epoch seconds
_BAN_LAST_LOG = 0.0        # throttle ban log spam


def note_binance_ban(err):
    """Record an IP ban from a -1003 error. Extends the global cooldown."""
    global _BAN_UNTIL_TS
    import re as _re
    m = _re.search(r'banned until (\d+)', str(err))
    if m:
        until = int(m.group(1)) / 1000.0 + 60.0   # +60s safety buffer
    else:
        until = time.time() + 600.0                # unknown window: 10 min
    if until > _BAN_UNTIL_TS:
        _BAN_UNTIL_TS = until
        remaining = max(0.0, until - time.time())
        log.error(f"🚫 BINANCE IP BAN detected — going silent for "
                  f"{remaining/3600:.1f}h (until "
                  f"{datetime.fromtimestamp(until, tz=timezone.utc).isoformat()})")


def binance_banned_now():
    return time.time() < _BAN_UNTIL_TS


def ban_remaining_secs():
    return max(0.0, _BAN_UNTIL_TS - time.time())


def log_ban_throttled():
    """Log ban status at most once per 10 minutes."""
    global _BAN_LAST_LOG
    if time.time() - _BAN_LAST_LOG > 600:
        _BAN_LAST_LOG = time.time()
        log.warning(f"🚫 IP banned by Binance — sleeping, "
                    f"{ban_remaining_secs()/3600:.1f}h remaining")


def is_rate_limit_error(e):
    try:
        return isinstance(e, BinanceAPIException) and int(getattr(e, 'code', 0)) == -1003
    except Exception:
        return False


# ── Disabled symbols persistence ─────────────────────────────────────────────
# SAFETY FIX: disabled symbols survive restarts — cannot re-enable by restarting

def load_disabled_symbols():
    """Load disabled symbols from disk. Returns a set."""
    if os.path.exists(DISABLED_SYMBOLS_PATH):
        try:
            with open(DISABLED_SYMBOLS_PATH) as f:
                data = json.load(f)
            result = set(data) if isinstance(data, list) else set()
            if result:
                log.warning(f"⚠️  Loaded disabled symbols from disk: {result}")
            return result
        except (json.JSONDecodeError, ValueError, TypeError):
            log.warning(f"⚠️  Corrupted {DISABLED_SYMBOLS_PATH} — treating as empty")
            return set()
    return set()


def save_disabled_symbols(disabled):
    """Persist disabled symbols to disk."""
    try:
        with open(DISABLED_SYMBOLS_PATH, "w") as f:
            json.dump(sorted(disabled), f)
    except Exception as e:
        log.error(f"❌ Failed to save disabled symbols: {e}")


# ── Exchange position count ──────────────────────────────────────────────────
# SAFETY FIX: count real exchange positions, not just local state

def count_exchange_open_positions(client, symbols):
    """Query exchange for actual open positions across all symbols."""
    count = 0
    for sym in symbols:
        try:
            pos_info = client.futures_position_information(symbol=sym)
            if abs(float(pos_info[0]['positionAmt'])) > 0:
                count += 1
        except Exception as e:
            log.error(f"❌ Could not check position for {sym}: {e}")
            # SAFETY FIX: if we can't check, assume occupied (conservative)
            count += 1
    return count


# ── Data fetch (futures klines — works from geo-blocked regions like Railway) ─
def fetch_mtf(client, symbol="BTCUSDT", bars=200):
    def fetch_tf(interval, limit):
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'ts','open','high','low','close','volume',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df[['ts','open','high','low','close','volume']].set_index('ts')

    df_1h  = fetch_tf(Client.KLINE_INTERVAL_1HOUR,  bars)
    df_4h  = fetch_tf(Client.KLINE_INTERVAL_4HOUR,  bars)
    df_1d  = fetch_tf(Client.KLINE_INTERVAL_1DAY,   bars)
    df_1w  = fetch_tf(Client.KLINE_INTERVAL_1WEEK,  bars)

    df_4h_ff = df_4h.reindex(df_1h.index, method='ffill').add_prefix('h4_')
    df_1d_ff = df_1d.reindex(df_1h.index, method='ffill').add_prefix('d1_')
    df_1w_ff = df_1w.reindex(df_1h.index, method='ffill').add_prefix('w1_')

    df = pd.concat([df_1h, df_4h_ff, df_1d_ff, df_1w_ff], axis=1).dropna()
    df = df.reset_index().rename(columns={'ts': 'Datetime'})
    return df

# ── Futures symbol info ───────────────────────────────────────────────────────
def get_futures_symbol_info(client, symbol):
    info = client.futures_exchange_info()
    sym = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
    if not sym:
        raise RuntimeError(f"Symbol {symbol} not found on futures")
    lot_filter   = next(f for f in sym['filters'] if f['filterType'] == 'LOT_SIZE')
    price_filter = next(f for f in sym['filters'] if f['filterType'] == 'PRICE_FILTER')
    step_size      = float(lot_filter['stepSize'])
    min_qty        = float(lot_filter['minQty'])
    tick_size      = float(price_filter['tickSize'])
    qty_precision  = sym.get('quantityPrecision', 3)
    price_precision = sym.get('pricePrecision', 2)
    return {
        "step_size": step_size,
        "min_qty": min_qty,
        "qty_prec": qty_precision,
        "tick_size": tick_size,
        "price_prec": price_precision,
    }

def round_qty(qty, step_size, precision):
    rounded = float(int(qty / step_size) * step_size)
    return round(rounded, precision)

def round_price(price, tick_size, price_precision):
    """Round to the symbol's tick size — required for ETH/SOL where tick != 0.1."""
    rounded = round(price / tick_size) * tick_size
    return round(rounded, price_precision)

# ── Order placement ───────────────────────────────────────────────────────────
def place_futures_order(client, symbol, side, qty, reduce_only=False):
    try:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        order = client.futures_create_order(**params)
        log.info(f"✅ [{symbol}] Order: {side} {qty} (reduceOnly={reduce_only})")
        return order
    except BinanceAPIException as e:
        log.error(f"❌ [{symbol}] Order failed: {e}")
        return None

def open_long(client, symbol, notional, price, step, min_qty, prec):
    qty = round_qty(notional / price, step, prec)
    if qty < min_qty:
        log.warning(f"[{symbol}] Qty {qty} < min {min_qty} — skip")
        return None, 0.0
    order = place_futures_order(client, symbol, "BUY", qty, reduce_only=False)
    return order, qty

def open_short(client, symbol, notional, price, step, min_qty, prec):
    qty = round_qty(notional / price, step, prec)
    if qty < min_qty:
        log.warning(f"[{symbol}] Qty {qty} < min {min_qty} — skip")
        return None, 0.0
    order = place_futures_order(client, symbol, "SELL", qty, reduce_only=False)
    return order, qty

def close_long(client, symbol, qty):
    return place_futures_order(client, symbol, "SELL", qty, reduce_only=True)

def close_short(client, symbol, qty):
    return place_futures_order(client, symbol, "BUY", qty, reduce_only=True)


def confirm_order_fill(client, symbol, order, requested_qty=0.0, tag=""):
    """Query order status to confirm fill. Returns (filled, avg_price, exec_qty) or
    (False, 0, 0) if not filled. Retries up to 3 times with short waits."""
    order_id = order.get('orderId')
    if not order_id:
        log.critical(f"{tag} No orderId in order response — cannot confirm fill")
        return False, 0.0, 0.0

    for attempt in range(3):
        try:
            info = client.futures_get_order(symbol=symbol, orderId=order_id)
            status = info.get('status', '')
            exec_qty = float(info.get('executedQty', 0))
            # avgPrice is the VWAP of the fill
            avg_price = float(info.get('avgPrice', 0))

            if status == 'FILLED':
                log.info(f"{tag} Order {order_id} FILLED | avgPrice=${avg_price:,.2f} | qty={exec_qty}")
                return True, avg_price, exec_qty
            elif status == 'PARTIALLY_FILLED':
                log.warning(f"{tag} Order {order_id} PARTIALLY_FILLED | "
                            f"avgPrice=${avg_price:,.2f} | execQty={exec_qty}")
                log.warning(f"{tag} PARTIAL FILL: requested {requested_qty} got {exec_qty} "
                            f"— SL will close full position via closePosition=true")
                return True, avg_price, exec_qty
            elif status in ('NEW', 'PENDING_NEW'):
                if attempt < 2:
                    time.sleep(1)
                    continue
                log.critical(f"{tag} Order {order_id} still {status} after 3 checks — treating as unfilled")
                return False, 0.0, 0.0
            else:
                # CANCELED, REJECTED, EXPIRED, etc.
                log.critical(f"{tag} Order {order_id} status={status} — NOT filled")
                return False, 0.0, 0.0
        except Exception as e:
            log.error(f"{tag} confirm_order_fill attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(1)

    log.critical(f"{tag} Could not confirm order {order_id} after 3 attempts")
    return False, 0.0, 0.0


# SAFETY FIX: emergency close now confirms fill and retries once
def emergency_close_position(client, symbol, position, qty, tag=""):
    """Market-close a position immediately. Returns (success, avg_price, exec_qty).
    Retries once if first close is not confirmed."""
    for attempt in range(2):  # SAFETY FIX: attempt + 1 retry
        if position == 1:
            order = close_long(client, symbol, qty)
        elif position == -1:
            order = close_short(client, symbol, qty)
        else:
            return False, 0.0, 0.0

        if order:
            # SAFETY FIX: confirm the emergency close fill
            filled, avg_price, exec_qty = confirm_order_fill(
                client, symbol, order, requested_qty=qty,
                tag=f"{tag} EMERGENCY_CLOSE")
            if filled:
                log.critical(f"{tag} EMERGENCY CLOSE confirmed for {symbol} | "
                             f"qty={exec_qty} @ ${avg_price:,.2f}")
                return True, avg_price, exec_qty
            if attempt == 0:
                log.critical(f"{tag} EMERGENCY CLOSE not confirmed — retrying once")
                time.sleep(2)
                continue
        else:
            if attempt == 0:
                log.critical(f"{tag} EMERGENCY CLOSE order failed — retrying once")
                time.sleep(2)
                continue

    log.critical(f"{tag} EMERGENCY CLOSE FAILED for {symbol} after 2 attempts "
                 f"— MANUAL INTERVENTION REQUIRED")
    return False, 0.0, 0.0


def safe_place_sl_or_exit(client, symbol, sl_side, sl_price, position, qty,
                          disabled_symbols, tag="", state=None):  # SL_SOFTWARE
    """SL_SOFTWARE: testnet doesn't support STOP_MARKET, manage SL in software.
    Stores the SL price in state for the cycle loop to check on each bar.
    The unused exchange-related parameters are kept for call-site compatibility."""  # SL_SOFTWARE
    if state is not None:  # SL_SOFTWARE
        state['sl_price'] = float(sl_price)  # SL_SOFTWARE
        log.info(f"{tag} SL stored in software @ ${sl_price} (side={sl_side})")  # SL_SOFTWARE
    else:  # SL_SOFTWARE
        log.warning(f"{tag} SL_SOFTWARE: state not passed — SL price NOT persisted")  # SL_SOFTWARE
    return True  # SL_SOFTWARE


# ── Helper: get SL fill price from recent orders ────────────────────────────
# SAFETY FIX: try to recover actual fill price when SL fired on exchange

def get_sl_fill_price(client, symbol, tag=""):
    """Try to find the most recent STOP_MARKET fill price for a symbol.
    Returns the avgPrice if found, or None."""
    try:
        recent_orders = client.futures_get_all_orders(symbol=symbol, limit=10)
        for o in reversed(recent_orders):
            if (o.get('status') == 'FILLED' and
                    o.get('type') in ('STOP_MARKET', 'STOP')):
                avg = float(o.get('avgPrice', 0))
                if avg > 0:
                    log.info(f"{tag} Found SL fill: orderId={o['orderId']} "
                             f"avgPrice=${avg:,.2f}")
                    return avg
    except Exception as e:
        log.error(f"{tag} Could not query recent orders for SL fill: {e}")
    return None


# ── PROB_GATE: extract policy action probabilities ───────────────────────────

def get_action_probs(model, obs):
    """Return PPO policy action probabilities [P(HOLD),P(LONG),P(SHORT),P(CLOSE)]
    for a single observation, or None on failure. This is the model's real
    conviction — deterministic argmax alone discards it."""
    try:
        import torch
        with torch.no_grad():
            t, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(t)
            return dist.distribution.probs.cpu().numpy().flatten()
    except Exception as e:
        log.warning(f"get_action_probs failed: {e}")
        return None


# ── SHADOW_LOG: record EVERY model signal regardless of gate outcome ─────────
# Purpose: build a real dataset to tune the gate on data instead of
# hand-tuned constants. One row per new-bar decision per symbol.

# Persistent dir (Railway volume) via SHADOW_DATA_DIR; falls back to HERE.
SHADOW_DIR = os.environ.get("SHADOW_DATA_DIR", HERE)


def shadow_log_path_for(symbol):
    os.makedirs(SHADOW_DIR, exist_ok=True)
    return os.path.join(SHADOW_DIR, f"shadow_log_{symbol.lower()}.csv")


def log_shadow_signal(symbol, bar_ts, action, model_probs, verdict, confidence,
                      rule_conf, tier, conditions, price):
    try:
        import csv
        path = shadow_log_path_for(symbol)
        new_file = not os.path.exists(path)
        probs = list(model_probs) if model_probs is not None else [None] * 4
        cond = conditions or {}
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["ts_utc", "bar_ts", "action",
                            "p_hold", "p_long", "p_short", "p_close",
                            "verdict", "confidence", "rule_conf", "tier",
                            "regime", "trend", "momentum", "volume", "session",
                            "price"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), bar_ts, int(action),
                *[f"{float(p):.4f}" if p is not None else "" for p in probs],
                verdict or "", f"{float(confidence):.4f}",
                f"{float(rule_conf):.4f}" if rule_conf is not None else "",
                tier or "",
                cond.get('regime', ''), cond.get('trend', ''),
                cond.get('momentum', ''), cond.get('volume', ''),
                cond.get('session', ''),
                f"{float(price):.4f}" if price else "",
            ])
    except Exception as e:
        log.warning(f"[{symbol[:3]}] shadow log write failed: {e}")


# ── Multi-agent orchestrator helpers ──────────────────────────────────────────

def orchestrator_gate(symbol, action, conditions, perception, memory,
                      state, narrative, ctx, log, tag, model_probs=None,
                      bar_ts=None):
    """
    Multi-agent orchestrated gate. Replaces the old single reasoning gate.
    Returns (verdict, confidence, reasoning_text, tier, sentiment_signal, audit)
    or None if we should skip this cycle.
    model_probs: PPO policy action probabilities — primary confidence signal.
    """
    sentiment_agent  = ctx.get('sentiment_agent')
    position_auditor = ctx.get('position_auditor')

    # ── Agent 1: Sentiment ───────────────────────────────────────────
    sentiment_signal = None
    if sentiment_agent:
        try:
            sentiment_signal = sentiment_agent.get_signal(symbol)
        except Exception as e:
            log.warning(f"{tag} Sentiment fetch failed: {e}")

    # ── Agent 2: Reasoning Engine (fixed) ────────────────────────────
    verdict, confidence, reasoning_text, tier = decide(
        action, conditions, perception, memory, narrative,
        sentiment_signal=sentiment_signal, model_probs=model_probs
    )

    log.info(f"\n{'='*55}\n{tag} {reasoning_text}\n{'='*55}")

    # ── SHADOW_LOG: record every decision, executed or not ────────────
    log_shadow_signal(
        symbol, bar_ts, action, model_probs, verdict, confidence,
        rule_conf=None, tier=tier, conditions=conditions,
        price=perception.get('price', 0.0) if perception else 0.0,
    )

    # ── Agent 3: Position Auditor (only when in trade) ────────────────
    if state.get('position', 0) != 0 and position_auditor:
        try:
            audit = position_auditor.audit(
                state, conditions, perception, sentiment_signal
            )
            log.info(f"{tag} [AUDITOR] {audit['action']} | score={audit['score']:.2f} | {audit['reason']}")

            if audit['action'] == 'EXIT_NOW':
                log.warning(f"{tag} 🔍 AUDITOR EXIT_NOW: {audit['reason']}")
                return ('AUDITOR_EXIT', confidence, reasoning_text, tier,
                        sentiment_signal, audit)

        except Exception as e:
            log.error(f"{tag} Position auditor failed: {e}")

    # ── Agent 4: Hard reject from reasoning engine ────────────────────
    if verdict == 'HARD_REJECT':
        sl_pct_rej, tp_pct_rej = _get_rejection_sl_tp(conditions, memory)
        _store_rejection(state, action, conditions, ctx, sl_pct_rej, tp_pct_rej)
        log.info(f"{tag} ⛔ HARD_REJECT — regime/session direction block")
        return None   # skip cycle

    # ── Standard reject ───────────────────────────────────────────────
    if action in (1, 2) and verdict not in ("EXECUTE", "WEAK_EXECUTE"):
        sl_pct_rej, tp_pct_rej = _get_rejection_sl_tp(conditions, memory)
        _store_rejection(state, action, conditions, ctx, sl_pct_rej, tp_pct_rej)
        log.info(f"{tag} ⛔ Trade REJECTED (confidence={confidence:.0%}) — stored for regret review")
        return None   # skip cycle

    return (verdict, confidence, reasoning_text, tier, sentiment_signal, None)


def _get_rejection_sl_tp(conditions, memory):
    return get_dynamic_sl_tp(conditions, memory)


def _store_rejection(state, action, conditions, ctx, sl_pct, tp_pct):
    """Store rejection in state for regret review."""
    state['last_rejected_action']     = action
    state['last_rejected_conditions'] = conditions
    state['last_rejected_sl_pct']     = sl_pct
    state['last_rejected_tp_pct']     = tp_pct


def _execute_entry(symbol, ctx, state, current_price, filters, tag, memory,
                   action, verdict, confidence, tier, conditions,
                   reasoning_text, signal_bar_ts):
    """
    Place an entry order for an approved signal. Used by both:
      - Direct 1H bar entries (legacy path)
      - 15m confirmation entries (new path)
    Returns True iff an entry was placed and confirmed filled.
    Mutates `state` and persists it.
    """
    exec_client      = ctx['exec_client']
    disabled_symbols = ctx['disabled_symbols']
    state_path       = state_path_for(symbol)

    step_size  = filters['step_size']
    min_qty    = filters['min_qty']
    qty_prec   = filters['qty_prec']
    tick_size  = filters['tick_size']
    price_prec = filters['price_prec']

    # Daily trade limit (per symbol, UTC day)
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get('last_trade_date') != today:
        state['daily_trades'] = 0
        state['last_trade_date'] = today
    if state['daily_trades'] >= MAX_DAILY_TRADES:
        log.warning(f"{tag} ⏸️  Daily trade limit reached "
                    f"({state['daily_trades']}/{MAX_DAILY_TRADES}) — skip entry")
        return False

    # Idempotency: never re-enter on the same originating signal bar
    trade_id = f"{symbol}_{signal_bar_ts}"
    if trade_id == state.get('last_trade_id'):
        log.warning(f"{tag} Duplicate signal — trade_id={trade_id} — skipping")
        return False

    # Exposure cap from exchange
    open_count = count_exchange_open_positions(exec_client, SYMBOLS)
    if open_count >= MAX_CONCURRENT_POSITIONS:
        log.warning(f"{tag} Exposure limit: {open_count}/{MAX_CONCURRENT_POSITIONS} "
                    f"exchange positions open — blocking new entry")
        return False

    sl_pct, tp_pct = get_dynamic_sl_tp(conditions, memory)

    # Risk-based sizing
    pre_trade_balance = fetch_live_balance(exec_client)
    if pre_trade_balance is None or pre_trade_balance <= 0:
        log.error(f"{tag} Balance fetch returned {pre_trade_balance} — aborting entry")
        return False
    risk_pct = RISK_PER_TRADE_BY_TIER.get(tier, 0.0)
    if risk_pct == 0.0:
        log.warning(f"{tag} ⛔ {tier} tier — risk_pct=0, skipping entry")
        return False
    risk_per_trade = pre_trade_balance * risk_pct
    risk_notional  = risk_per_trade / max(sl_pct, 0.001)
    trade_notional = min(risk_notional, TRADE_NOTIONAL,
                         pre_trade_balance * LEVERAGE * 0.30)
    if trade_notional / max(current_price, 1) < min_qty:
        log.warning(f"{tag} Risk-sized notional ${trade_notional:.2f} too small "
                    f"for min_qty={min_qty} — skipping entry")
        return False

    if trade_notional / max(current_price, 1) < min_qty:
        log.warning(f"{tag} Tier-sized notional ${trade_notional:.2f} too small "
                    f"for min_qty={min_qty} — skipping entry")
        return False

    log.info(f"{tag} 📐 Tier sizing: balance=${pre_trade_balance:,.2f} | "
             f"Tier={tier} | risk={risk_pct:.1%} (${risk_per_trade:.2f}) | "
             f"SL={sl_pct:.1%} | raw_notional=${risk_notional:,.2f} | "
             f"capped=${trade_notional:,.2f} | TP={tp_pct:.1%} | "
             f"WR={memory.get_win_rate(conditions)}")

    if action == 1:  # LONG
        order, qty = open_long(exec_client, symbol, trade_notional, current_price,
                               step_size, min_qty, qty_prec)
        if not order:
            return False
        filled, avg_price, exec_qty = confirm_order_fill(
            exec_client, symbol, order, requested_qty=qty, tag=tag)
        if not filled:
            log.critical(f"{tag} LONG order NOT confirmed filled — state NOT updated")
            return False
        actual_qty   = exec_qty if exec_qty > 0 else qty
        actual_price = avg_price if avg_price > 0 else current_price

        state['position']         = 1
        state['entry_price']      = actual_price
        state['entry_time']       = datetime.now(timezone.utc).isoformat()
        state['qty']              = actual_qty
        state['trade_count']     += 1
        state['daily_trades']    += 1
        state['sl_pct']           = sl_pct
        state['tp_pct']           = tp_pct
        state['entry_conditions'] = conditions
        state['entry_confidence'] = confidence
        state['entry_verdict']    = verdict
        state['entry_reasoning']  = (reasoning_text or "")[:300]
        state['last_trade_id']    = trade_id
        state['breakeven_set']    = False
        state['trail_1r_set']     = False
        save_state(state, state_path)
        log.info(f"{tag} 📈 LONG opened | qty={actual_qty} @ avg=${actual_price:,.2f}")
        sl_price = round_price(actual_price * (1 - sl_pct), tick_size, price_prec)
        log_trade_event(symbol, "OPEN", "LONG", actual_qty, actual_price,
                        trade_id=trade_id, sl_price=sl_price, tp_pct=tp_pct,
                        confidence=confidence, verdict=verdict)
        sl_ok = safe_place_sl_or_exit(
            exec_client, symbol, "SELL", sl_price,
            state['position'], actual_qty, disabled_symbols, tag, state=state)
        save_state(state, state_path)
        if not sl_ok:
            state['position']    = 0
            state['entry_price'] = 0.0
            state['entry_time']  = None
            state['qty']         = 0.0
            save_state(state, state_path)
            return False
        return True

    if action == 2:  # SHORT
        order, qty = open_short(exec_client, symbol, trade_notional, current_price,
                                step_size, min_qty, qty_prec)
        if not order:
            return False
        filled, avg_price, exec_qty = confirm_order_fill(
            exec_client, symbol, order, requested_qty=qty, tag=tag)
        if not filled:
            log.critical(f"{tag} SHORT order NOT confirmed filled — state NOT updated")
            return False
        actual_qty   = exec_qty if exec_qty > 0 else qty
        actual_price = avg_price if avg_price > 0 else current_price

        state['position']         = -1
        state['entry_price']      = actual_price
        state['entry_time']       = datetime.now(timezone.utc).isoformat()
        state['qty']              = actual_qty
        state['trade_count']     += 1
        state['daily_trades']    += 1
        state['sl_pct']           = sl_pct
        state['tp_pct']           = tp_pct
        state['entry_conditions'] = conditions
        state['entry_confidence'] = confidence
        state['entry_verdict']    = verdict
        state['entry_reasoning']  = (reasoning_text or "")[:300]
        state['last_trade_id']    = trade_id
        state['breakeven_set']    = False
        state['trail_1r_set']     = False
        save_state(state, state_path)
        log.info(f"{tag} 📉 SHORT opened | qty={actual_qty} @ avg=${actual_price:,.2f}")
        sl_price = round_price(actual_price * (1 + sl_pct), tick_size, price_prec)
        log_trade_event(symbol, "OPEN", "SHORT", actual_qty, actual_price,
                        trade_id=trade_id, sl_price=sl_price, tp_pct=tp_pct,
                        confidence=confidence, verdict=verdict)
        sl_ok = safe_place_sl_or_exit(
            exec_client, symbol, "BUY", sl_price,
            state['position'], actual_qty, disabled_symbols, tag, state=state)
        save_state(state, state_path)
        if not sl_ok:
            state['position']    = 0
            state['entry_price'] = 0.0
            state['entry_time']  = None
            state['qty']         = 0.0
            save_state(state, state_path)
            return False
        return True

    return False


def fetch_15m_df(client, symbol, limit=20):
    """Fetch the last `limit` 15m futures candles as a DataFrame with float OHLC."""
    klines = client.futures_klines(symbol=symbol, interval='15m', limit=limit)
    df = pd.DataFrame(klines, columns=[
        'ts','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df


def check_15m_confirmation(df_15m, action, tier=None):
    """
    PULLBACK_FIX: Validate a 1H-approved signal against the 15m timeframe.
    Returns (ok: bool, reason: str).

    The old filter required 3 consecutive 15m closes IN the signal direction —
    i.e. it forced every entry into a momentum-chase profile. Seed data showed
    the model's BEST setups were dip entries (STRONG_DOWN buckets: 72-76% WR)
    and its WORST were chase entries (WEAK_UP buckets). The old filter selected
    for exactly the losing profile and structurally blocked shorts too.

    New logic:
      a) A+/A tiers: confirmation waived entirely (high conviction — enter now)
      b) B/C tiers: PULLBACK entry — for LONG, price must have pulled back
         0.2%-2.0% from the recent 15m high (buying the dip, not the breakout).
         For SHORT, mirrored against the recent 15m low.
      c) Momentum-exhaustion guard kept: live 15m bar move from open <= 1%
      d) caller is responsible for the 'no open position' check
    """
    # (a) High-conviction tiers skip 15m confirmation
    if tier in ('A_PLUS', 'A'):
        return True, f"tier {tier} — 15m confirmation waived (high conviction)"

    if len(df_15m) < 10:
        return False, f"not enough 15m bars ({len(df_15m)})"

    live_open  = float(df_15m['open'].iloc[-1])
    live_close = float(df_15m['close'].iloc[-1])
    price      = live_close

    # (c) Momentum-exhaustion guard on the live (in-progress) 15m bar
    move_pct = abs(live_close - live_open) / max(live_open, 1e-9)
    if move_pct > 0.01:
        return False, f"15m momentum exhausted: {move_pct:.2%} from open"

    # (b) Pullback logic over the last 8 CLOSED 15m bars (exclude live bar)
    window = df_15m.iloc[-9:-1]
    recent_high = float(window['high'].max())
    recent_low  = float(window['low'].min())

    PULLBACK_MIN = 0.002   # 0.2% — enough of a dip to not be a breakout chase
    PULLBACK_MAX = 0.020   # 2.0% — beyond this the 1H thesis may be breaking

    if action == 1:  # LONG — want price pulled back from recent high
        pullback = (recent_high - price) / max(recent_high, 1e-9)
        if pullback < PULLBACK_MIN:
            return False, (f"no pullback yet: {pullback:.2%} below 15m high "
                           f"${recent_high:,.2f} — avoiding breakout chase")
        if pullback > PULLBACK_MAX:
            return False, (f"pullback too deep: {pullback:.2%} below 15m high — "
                           f"possible thesis break")
        return True, f"dip entry OK: {pullback:.2%} below recent 15m high"

    elif action == 2:  # SHORT — want price bounced up from recent low
        bounce = (price - recent_low) / max(recent_low, 1e-9)
        if bounce < PULLBACK_MIN:
            return False, (f"no bounce yet: {bounce:.2%} above 15m low "
                           f"${recent_low:,.2f} — avoiding breakdown chase")
        if bounce > PULLBACK_MAX:
            return False, (f"bounce too high: {bounce:.2%} above 15m low — "
                           f"possible thesis break")
        return True, f"bounce entry OK: {bounce:.2%} above recent 15m low"

    return False, f"unsupported action {action}"


# ── Per-symbol cycle logic ────────────────────────────────────────────────────

def process_symbol(symbol, ctx):
    """
    Run one bar-cycle of logic for a single symbol. Mutates ctx['states'][symbol]
    and ctx['last_bar_times'][symbol] in place. Catches all exceptions so one
    bad symbol can't stall the others.
    """
    disabled_symbols = ctx['disabled_symbols']
    if symbol in disabled_symbols:
        return  # symbol killed — skip entirely

    state         = ctx['states'][symbol]
    memory        = ctx['memories'][symbol]
    filters       = ctx['filters'][symbol]
    data_client   = ctx['data_client']
    exec_client   = ctx['exec_client']
    model         = ctx['model']
    max_loss_per_trade = ctx['max_loss_per_trade']

    step_size  = filters['step_size']
    min_qty    = filters['min_qty']
    qty_prec   = filters['qty_prec']
    tick_size  = filters['tick_size']
    price_prec = filters['price_prec']

    state_path = state_path_for(symbol)
    tag = f"[{symbol[:3]}]"   # short tag for log readability

    try:
        # ── RATE_LIMIT: full silence while IP-banned ──────────────────────────
        if binance_banned_now():
            return

        # ── RATE_LIMIT: clock-gated kline fetching ────────────────────────────
        # Old behavior: fetch 4 timeframes × klines EVERY 60s cycle for every
        # symbol (~15 req/min continuously) even with nothing to do. New:
        # klines are only fetched when a new closed 1H bar can exist that we
        # haven't processed. Idle cycles (flat, no pending signal) make ZERO
        # API calls; holding/pending cycles fetch only the mark price.
        now_utc    = datetime.now(timezone.utc)
        hour_floor = now_utc.replace(minute=0, second=0, microsecond=0)
        # last closed 1H bar's OPEN time (bar opening at 14:00 closes at 15:00)
        expected_closed = hour_floor - timedelta(hours=1)
        grace_ok   = (now_utc - hour_floor).total_seconds() >= 20  # let bar data settle
        last_seen  = ctx['last_bar_times'][symbol]
        may_have_new_bar = grace_ok and (
            last_seen is None or pd.Timestamp(last_seen) != pd.Timestamp(expected_closed))

        has_position = state.get('position', 0) != 0
        has_pending  = bool(state.get('pending_signal'))

        if not may_have_new_bar and not has_position and not has_pending:
            return  # RATE_LIMIT: truly idle — zero API calls this cycle

        df = None
        new_bar = False
        latest_bar_time = last_seen
        if may_have_new_bar:
            df = fetch_mtf(data_client, symbol, bars=200)
            if len(df) < MIN_BARS:
                log.warning(f"{tag} Not enough bars ({len(df)} < {MIN_BARS})")
                return
            latest_bar_time = df['Datetime'].iloc[-2]
            new_bar = latest_bar_time != ctx['last_bar_times'][symbol]

        # SL_SOFTWARE: fetch live mark price so SL/TP/protection checks react
        # to real-time moves. RATE_LIMIT: this is now the ONLY per-minute call.
        try:
            mark_price = float(data_client.futures_mark_price(symbol=symbol)['markPrice'])
        except Exception as _mpe:
            if is_rate_limit_error(_mpe):
                note_binance_ban(_mpe)
                return
            if df is not None:
                mark_price = float(df['close'].iloc[-2])
                log.warning(f"{tag} mark_price fetch failed ({_mpe}) — falling back to bar close ${mark_price:,.2f}")
            else:
                log.warning(f"{tag} mark_price fetch failed ({_mpe}) and no klines this cycle — skipping")
                return
        current_price = mark_price  # used for SL/TP/protection + entry sizing

        # Nothing to manage when no new bar AND no open position
        if not new_bar and state.get('position', 0) == 0 and not has_pending:
            return

        if new_bar:
            ctx['last_bar_times'][symbol] = latest_bar_time
            bar_close = float(df['close'].iloc[-2])
            log.info(f"\n{tag} 📊 New bar: {latest_bar_time} | Close: ${bar_close:,.2f} | Mark: ${mark_price:,.2f}")

        # ── Regret review (BAR-GATED — uses df indexing) ─────────────────────
        if new_bar and (state.get('last_rejected_action') is not None and state.get('position') == 0):
            try:
                rejected_ts = state.get('last_rejected_bar_ts')
                rejected_action = state['last_rejected_action']
                rejected_conditions = state['last_rejected_conditions']
                r_sl = state.get('last_rejected_sl_pct', 0.015)
                r_tp = state.get('last_rejected_tp_pct', 0.04)

                rejected_idx = None
                if rejected_ts:
                    target = pd.Timestamp(rejected_ts)
                    matches = df.index[df['Datetime'] == target].tolist()
                    if matches:
                        rejected_idx = int(matches[0])

                if rejected_idx is not None and rejected_idx < len(df) - 1:
                    was_profitable, pnl, exit_reason = simulate_missed_trade(
                        df, rejected_idx, rejected_action, r_sl, r_tp
                    )
                    if was_profitable is not None:
                        memory.record_missed_trade(rejected_conditions, was_profitable)
                        action_name = "BUY" if rejected_action == 1 else "SELL"
                        outcome = "✅ WOULD HAVE WON" if was_profitable else "❌ WOULD HAVE LOST"
                        log.info(f"{tag} 🧠 Regret: rejected {action_name} → {outcome} ({pnl*100:+.2f}% via {exit_reason})")
                else:
                    log.warning(f"{tag} ⚠️  Regret review: rejected bar {rejected_ts} not in window — skipping")
            except Exception as e:
                log.error(f"{tag} ❌ Regret review failed: {e}")
            finally:
                state['last_rejected_action'] = None
                state['last_rejected_conditions'] = None
                state['last_rejected_bar_ts'] = None
                save_state(state, state_path)

        # ── Position reconcile ───────────────────────────────────────────────
        try:
            pos_info = exec_client.futures_position_information(symbol=symbol)
            exchange_qty = float(pos_info[0]['positionAmt'])
            if exchange_qty == 0 and state['position'] != 0:
                log.warning(f"{tag} ⚠️  Exchange flat but state={state['position']} — SL likely fired, resyncing")

                # SAFETY FIX: try to get actual SL fill price instead of using current_price
                sl_fill_price = get_sl_fill_price(exec_client, symbol, tag)
                exit_price_for_pnl = sl_fill_price if sl_fill_price else current_price
                if not sl_fill_price:
                    log.warning(f"{tag} ⚠️  Could not find SL fill price — using current "
                                f"${current_price:,.2f} as estimate for PnL")

                pnl_pct = (exit_price_for_pnl - state['entry_price']) / state['entry_price'] * state['position']
                pnl_usdt = pnl_pct * TRADE_NOTIONAL
                state['total_pnl_usdt'] += pnl_usdt
                if pnl_pct > 0:
                    state['wins'] += 1
                    state['consecutive_losses'] = 0
                else:
                    state['losses'] += 1
                    state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1
                    if state['consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
                        log.critical(f"{tag} {symbol} HIT {MAX_CONSECUTIVE_LOSSES} CONSECUTIVE LOSSES — DISABLING SYMBOL")
                        disabled_symbols.add(symbol)
                        save_disabled_symbols(disabled_symbols)  # SAFETY FIX: persist
                if state.get('entry_conditions'):
                    try:
                        memory.record_trade(
                            conditions=state['entry_conditions'],
                            action="BUY" if state['position'] == 1 else "SELL",
                            entry_price=state['entry_price'],
                            exit_price=exit_price_for_pnl,  # SAFETY FIX: use best available price
                            pnl_pct=pnl_pct,
                            pnl_usdt=pnl_usdt,
                            confidence=state.get('entry_confidence', 0.5),
                            verdict=state.get('entry_verdict', 'UNKNOWN'),
                            reasoning=state.get('entry_reasoning', '') + " [resynced]",
                            trade_type="live"
                        )
                    except Exception as e:
                        log.error(f"{tag} ❌ Memory record failed: {e}")
                log_trade_event(  # SAFETY FIX: log reconcile close event
                    symbol, "CLOSE", "LONG" if state['position'] == 1 else "SHORT",
                    state['qty'], exit_price_for_pnl,
                    trade_id=state.get('last_trade_id', ''),
                    pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                    close_reason="SL_RECONCILE",
                    confidence=state.get('entry_confidence', 0.0),
                    verdict=state.get('entry_verdict', ''))
                state['position'] = 0
                state['entry_price'] = 0.0
                state['entry_time'] = None
                state['qty'] = 0.0
                state['entry_conditions'] = None
                state['entry_confidence'] = 0.5
                state['entry_verdict'] = None
                state['entry_reasoning'] = ""
                state['breakeven_set'] = False   # SAFETY FIX: reset trailing flags
                state['trail_1r_set'] = False     # SAFETY FIX: reset trailing flags
                state['sl_price'] = 0.0  # SL_SOFTWARE: reset SL on close
                save_state(state, state_path)
            elif exchange_qty != 0 and state['position'] == 0:
                # ── Auto-rebuild state from exchange data ────────────────
                log.critical(f"{tag} Exchange has qty={exchange_qty} but state=flat — rebuilding state from exchange")
                exchange_entry_price = float(pos_info[0].get('entryPrice', 0))

                # SAFETY FIX: validate exchange data before rebuilding
                if abs(exchange_qty) < min_qty:
                    log.critical(f"{tag} Exchange qty {exchange_qty} below min_qty {min_qty} "
                                 f"— dust position, skipping rebuild")
                    # Don't rebuild for dust — it can't be closed normally
                else:
                    if exchange_qty > 0:
                        state['position'] = 1
                        state['qty'] = exchange_qty
                    else:
                        state['position'] = -1
                        state['qty'] = abs(exchange_qty)

                    # SAFETY FIX: warn if entryPrice is missing/zero
                    if exchange_entry_price <= 0:
                        log.warning(f"{tag} ⚠️  Exchange entryPrice is {exchange_entry_price} "
                                    f"— using current_price ${current_price:,.2f} as fallback")
                    state['entry_price'] = exchange_entry_price if exchange_entry_price > 0 else current_price
                    if not state.get('entry_time'):
                        state['entry_time'] = datetime.now(timezone.utc).isoformat()
                        log.warning(f"{tag} entry_time missing after state rebuild — set to now (conservative)")
                    state['breakeven_set'] = False
                    state['trail_1r_set'] = False
                    state['sl_price'] = 0.0  # SL_SOFTWARE: rebuild SL fresh below
                    log.critical(f"{tag} State rebuilt: position={state['position']} | "
                                 f"entry=${state['entry_price']:,.2f} | qty={state['qty']}")
                    save_state(state, state_path)
                    # Place a fresh SL for the rebuilt position
                    rebuild_sl_pct = state.get('sl_pct', 0.015)
                    if state['position'] == 1:
                        rebuild_sl_side = "SELL"
                        rebuild_sl_price = round_price(
                            state['entry_price'] * (1 - rebuild_sl_pct), tick_size, price_prec)
                    else:
                        rebuild_sl_side = "BUY"
                        rebuild_sl_price = round_price(
                            state['entry_price'] * (1 + rebuild_sl_pct), tick_size, price_prec)
                    try:
                        exec_client.futures_cancel_all_open_orders(symbol=symbol)
                    except Exception as e:
                        log.error(f"{tag} Failed to cancel stale orders during rebuild: {e}")
                    safe_place_sl_or_exit(
                        exec_client, symbol, rebuild_sl_side, rebuild_sl_price,
                        state['position'], state['qty'], disabled_symbols, tag,
                        state=state)  # SL_SOFTWARE
                    save_state(state, state_path)  # SL_SOFTWARE: persist sl_price
                    # Continue cycle with rebuilt state (do NOT return)

            # SAFETY FIX: detect sign disagreement between local state and exchange
            elif exchange_qty != 0 and state['position'] != 0:
                exchange_side = 1 if exchange_qty > 0 else -1
                if exchange_side != state['position']:
                    log.critical(f"{tag} SIDE MISMATCH: local={state['position']} "
                                 f"exchange={exchange_side} (qty={exchange_qty}) "
                                 f"— closing exchange position and resetting state")
                    # Close the exchange position to resolve ambiguity
                    success, _, _ = emergency_close_position(
                        exec_client, symbol, exchange_side, abs(exchange_qty), tag)
                    state['position'] = 0
                    state['entry_price'] = 0.0
                    state['entry_time'] = None
                    state['qty'] = 0.0
                    state['entry_conditions'] = None
                    state['breakeven_set'] = False
                    state['trail_1r_set'] = False
                    save_state(state, state_path)
                    if not success:
                        disabled_symbols.add(symbol)
                        save_disabled_symbols(disabled_symbols)
                        log.critical(f"{tag} {symbol} DISABLED — could not resolve side mismatch")
                    return  # skip rest of cycle after mismatch resolution

        except Exception as e:
            log.error(f"{tag} ❌ Position reconcile failed: {e}")

        # ── Bar-gated decisions: model + reasoning + entry sizing ───────────
        # On no-new-bar cycles these are skipped; only position-management
        # below runs (so SL/TP can react to live mark price every minute).
        action = None
        entries_allowed = False
        verdict = None
        confidence = 0.5
        reasoning_text = ""
        tier = None
        sentiment_signal = None
        audit = None
        conditions = None
        narrative = None
        trade_notional = TRADE_NOTIONAL
        trade_id = None

        if new_bar:
            # ── Build observation ────────────────────────────────────────────
            env = CryptoTechEnv(df.iloc[:-1].reset_index(drop=True))  # NEW_MODEL: 44-feature env
            env.reset()
            obs = env._get_obs(len(df) - 2)

            in_trade = 1.0 if state['position'] != 0 else 0.0
            direction = float(state['position'])
            if state['position'] != 0 and state['entry_price'] > 0:
                upnl = (current_price - state['entry_price']) / state['entry_price'] * state['position']
                upnl_norm = float(np.clip(upnl * 10, -1, 1))
            else:
                upnl_norm = 0.0
            if state['entry_time']:
                entry_dt = datetime.fromisoformat(state['entry_time'])
                bars_held = int((datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600)
                bars_norm = float(np.clip(bars_held / 48, 0, 1))
            else:
                bars_norm = 0.0

            obs[27] = in_trade
            obs[28] = direction
            obs[29] = upnl_norm
            obs[30] = bars_norm

            # ── Model decision ───────────────────────────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            model_probs = get_action_probs(model, obs)  # PROB_GATE
            action_names = {0: "HOLD", 1: "BUY (long)", 2: "SELL (short)", 3: "CLOSE"}
            if model_probs is not None:
                log.info(f"{tag} 🤖 Decision: {action_names[action]} | "
                         f"P(H/L/S/C)={'/'.join(f'{p:.0%}' for p in model_probs)} | "
                         f"Position: {state['position']}")
            else:
                log.info(f"{tag} 🤖 Decision: {action_names[action]} | Position: {state['position']}")

            # ── Reasoning gate ───────────────────────────────────────────────
            perception = perceive(df, state)
            conditions, narrative = interpret(perception)

            bar_ts_for_log = (latest_bar_time.isoformat()
                              if hasattr(latest_bar_time, 'isoformat')
                              else str(latest_bar_time))
            result = orchestrator_gate(
                symbol, action, conditions, perception, memory,
                state, narrative, ctx, log, tag,
                model_probs=model_probs, bar_ts=bar_ts_for_log
            )
            if result is None:
                if state.get('last_rejected_action') is not None:
                    state['last_rejected_bar_ts'] = latest_bar_time.isoformat() if hasattr(latest_bar_time, 'isoformat') else str(latest_bar_time)
                    save_state(state, state_path)
                wr = state['wins'] / max(1, state['wins'] + state['losses'])
                log.info(f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")
                return

            verdict, confidence, reasoning_text, tier, sentiment_signal, audit = result

            # Handle auditor EXIT_NOW
            if verdict == 'AUDITOR_EXIT':
                action = 3
                state['_auditor_exit'] = True
                state['_auditor_reason'] = audit['reason'][:200]
            else:
                state.pop('_auditor_exit', None)

            # ── Daily trade limit gate (per symbol) ──────────────────────────
            today = datetime.now(timezone.utc).date().isoformat()
            if state.get('last_trade_date') != today:
                state['daily_trades'] = 0
                state['last_trade_date'] = today
            entries_allowed = state['daily_trades'] < MAX_DAILY_TRADES
            if not entries_allowed and state['position'] == 0 and action in (1, 2):
                log.warning(f"{tag} ⏸️  Daily trade limit reached ({state['daily_trades']}/{MAX_DAILY_TRADES}) — skip entry")

        # ── 1H signal storage: approve into pending_signal, defer execution ──
        # Actual entry is gated by the 15m confirmation block below.
        if new_bar and state['position'] == 0 and entries_allowed and action in (1, 2):
            if verdict in ('EXECUTE', 'WEAK_EXECUTE'):
                bar_ts_str = (latest_bar_time.isoformat()
                              if hasattr(latest_bar_time, 'isoformat')
                              else str(latest_bar_time))
                state['pending_signal'] = {
                    'action':         int(action),
                    'verdict':        verdict,
                    'confidence':     float(confidence),
                    'tier':           tier,
                    'bar_ts':         bar_ts_str,
                    'expires_at':     (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                    'conditions':     conditions,
                    'reasoning_text': (reasoning_text or "")[:300],
                }
                # Reset the 15m timer so the first check fires on the next cycle
                state['last_15m_check'] = None
                save_state(state, state_path)
                log.info(f"{tag} 🕒 1H signal approved (action={action}, verdict={verdict}, "
                         f"conf={confidence:.0%}, tier={tier}) — pending 15m confirmation, "
                         f"expires in 1h")

        # ── 15m entry confirmation: every 15+ min, check pending signal ─────
        if state['position'] == 0 and state.get('pending_signal'):
            pending = state['pending_signal']
            now = datetime.now(timezone.utc)

            try:
                expires_at = datetime.fromisoformat(pending['expires_at'])
            except Exception:
                expires_at = now  # treat malformed as expired

            if now >= expires_at:
                log.info(f"{tag} ⏰ Pending 1H signal expired without 15m confirmation — clearing")
                state['pending_signal'] = None
                save_state(state, state_path)
            else:
                last_15m_str = state.get('last_15m_check')
                last_15m_dt = None
                if last_15m_str:
                    try:
                        last_15m_dt = datetime.fromisoformat(last_15m_str)
                    except Exception:
                        last_15m_dt = None
                elapsed_min = ((now - last_15m_dt).total_seconds() / 60.0
                               if last_15m_dt else float('inf'))

                if elapsed_min >= 15:
                    state['last_15m_check'] = now.isoformat()
                    save_state(state, state_path)
                    try:
                        df_15m = fetch_15m_df(data_client, symbol, limit=20)
                        ok, why = check_15m_confirmation(df_15m, int(pending['action']),
                                                         tier=pending.get('tier'))
                        if ok:
                            log.info(f"{tag} ✅ 15m confirmation OK: {why}")
                            placed = _execute_entry(
                                symbol, ctx, state, current_price, filters, tag, memory,
                                action=int(pending['action']),
                                verdict=pending['verdict'],
                                confidence=float(pending['confidence']),
                                tier=pending['tier'],
                                conditions=pending['conditions'],
                                reasoning_text=pending.get('reasoning_text', ''),
                                signal_bar_ts=pending['bar_ts'],
                            )
                            if placed:
                                state['pending_signal'] = None
                                save_state(state, state_path)
                        else:
                            log.info(f"{tag} ⏳ 15m check failed: {why} — retry in 15m")
                    except Exception as e:
                        log.error(f"{tag} 15m confirmation block failed: {e}")

        # ── Position management ──────────────────────────────────────────────
        if state['position'] != 0:  # in a position
            sl_pct = state.get('sl_pct', 0.015)
            tp_pct = state.get('tp_pct', 0.04)
            entry = state['entry_price']
            force_close = False
            close_reason = None

            # Auditor exit takes priority over everything else
            if state.get('_auditor_exit'):
                force_close  = True
                close_reason = f"AUDITOR: {state.get('_auditor_reason', 'thesis broken')}"
                state.pop('_auditor_exit', None)
                state.pop('_auditor_reason', None)

            # HORIZON_FIX: time limit now matches the training horizon (48 bars)
            if not force_close and state.get('entry_time'):
                try:
                    entry_dt = datetime.fromisoformat(state['entry_time'])
                    held_seconds = (datetime.now(timezone.utc) - entry_dt).total_seconds()
                    if held_seconds > MAX_HOLD_HOURS * 3600:
                        force_close = True
                        close_reason = f"TIME_LIMIT: {MAX_HOLD_HOURS}h limit"
                        log.warning(f"{tag} ⏰ TIME LIMIT: position held "
                                    f"{held_seconds/3600:.1f}h > {MAX_HOLD_HOURS}h — force closing")
                except Exception as _te:
                    log.warning(f"{tag} Could not check entry time for 8h limit: {_te}")

            # Unrealized PnL (used for account protection + voluntary close gate)
            upnl_pct = (current_price - entry) / entry * state['position']

            # SL / TP checks — SL_SOFTWARE: prefer absolute sl_price from state,
            # fallback to entry * (1 ± sl_pct) if sl_price not set yet  # SL_SOFTWARE
            sl_price_state = state.get('sl_price', 0.0)  # SL_SOFTWARE
            if state['position'] == 1:
                sl_trigger = sl_price_state if sl_price_state > 0 else entry * (1 - sl_pct)  # SL_SOFTWARE
                if current_price <= sl_trigger:
                    log.info(f"{tag} 🛑 LONG SL hit @ ${current_price:,.2f} (trigger ${sl_trigger:,.2f})")  # SL_SOFTWARE
                    force_close = True
                    close_reason = "SL"
                elif current_price >= entry * (1 + tp_pct):
                    log.info(f"{tag} 🎯 LONG TP @ ${current_price:,.2f}")
                    force_close = True
                    close_reason = "TP"
            elif state['position'] == -1:
                sl_trigger = sl_price_state if sl_price_state > 0 else entry * (1 + sl_pct)  # SL_SOFTWARE
                if current_price >= sl_trigger:
                    log.info(f"{tag} 🛑 SHORT SL hit @ ${current_price:,.2f} (trigger ${sl_trigger:,.2f})")  # SL_SOFTWARE
                    force_close = True
                    close_reason = "SL"
                elif current_price <= entry * (1 - tp_pct):
                    log.info(f"{tag} 🎯 SHORT TP @ ${current_price:,.2f}")
                    force_close = True
                    close_reason = "TP"

            # Account protection
            if not force_close:
                upnl_usdt = upnl_pct * TRADE_NOTIONAL
                if upnl_usdt < 0 and abs(upnl_usdt) >= max_loss_per_trade:
                    log.warning(f"{tag} 🛡️ Account protection: loss ${abs(upnl_usdt):.2f} >= limit ${max_loss_per_trade:.2f}")
                    force_close = True
                    close_reason = "ACCOUNT_PROTECTION"

            # ── Breakeven: move SL to entry when price reaches 50% of TP distance ──
            if not force_close and not state.get('breakeven_set', False):
                be_triggered = False
                if state['position'] == 1 and current_price >= entry * (1 + tp_pct * 0.5):
                    be_triggered = True
                    be_side = "SELL"
                elif state['position'] == -1 and current_price <= entry * (1 - tp_pct * 0.5):
                    be_triggered = True
                    be_side = "BUY"

                if be_triggered:
                    state['sl_pct'] = 0.0
                    sl_pct = 0.0   # update local var so SL check uses breakeven if re-evaluated
                    state['breakeven_set'] = True
                    save_state(state, state_path)
                    log.info(f"{tag} 🔒 Breakeven set — SL moved to entry @ ${entry:,.2f}")
                    try:
                        exec_client.futures_cancel_all_open_orders(symbol=symbol)
                    except Exception as e:
                        log.error(f"{tag} Failed to cancel open orders: {e}")
                    be_price = round_price(entry, tick_size, price_prec)
                    sl_ok = safe_place_sl_or_exit(
                        exec_client, symbol, be_side, be_price,
                        state['position'], state['qty'], disabled_symbols, tag,
                        state=state)  # SL_SOFTWARE
                    save_state(state, state_path)  # SL_SOFTWARE: persist sl_price
                    if not sl_ok:
                        state['position'] = 0
                        state['entry_price'] = 0.0
                        state['entry_time'] = None
                        state['qty'] = 0.0
                        save_state(state, state_path)
                        return

            # ── Trail 1R: at 75% TP, move SL to 50% TP (locks in ~1R profit) ──
            if not force_close and state.get('breakeven_set', False) and not state.get('trail_1r_set', False):
                trail_triggered = False
                trail_sl_price = None
                if state['position'] == 1 and current_price >= entry * (1 + tp_pct * 0.75):
                    trail_triggered = True
                    trail_sl_price = round_price(entry * (1 + tp_pct * 0.5), tick_size, price_prec)
                    trail_side = "SELL"
                    # Update sl_pct so software SL check uses the new level
                    state['sl_pct'] = tp_pct * 0.5
                    sl_pct = state['sl_pct']
                elif state['position'] == -1 and current_price <= entry * (1 - tp_pct * 0.75):
                    trail_triggered = True
                    trail_sl_price = round_price(entry * (1 - tp_pct * 0.5), tick_size, price_prec)
                    trail_side = "BUY"
                    state['sl_pct'] = tp_pct * 0.5
                    sl_pct = state['sl_pct']

                if trail_triggered:
                    state['trail_1r_set'] = True
                    save_state(state, state_path)
                    log.info(f"{tag} 📈 Trail 1R set — SL moved to 50% TP @ ${trail_sl_price}")
                    try:
                        exec_client.futures_cancel_all_open_orders(symbol=symbol)
                    except Exception as e:
                        log.error(f"{tag} Failed to cancel open orders: {e}")
                    sl_ok = safe_place_sl_or_exit(
                        exec_client, symbol, trail_side, trail_sl_price,
                        state['position'], state['qty'], disabled_symbols, tag,
                        state=state)  # SL_SOFTWARE
                    save_state(state, state_path)  # SL_SOFTWARE: persist sl_price
                    if not sl_ok:
                        state['position'] = 0
                        state['entry_price'] = 0.0
                        state['entry_time'] = None
                        state['qty'] = 0.0
                        save_state(state, state_path)
                        return

            # ── Voluntary close gate ──
            # Blocked if profit < 50% TP (let breakeven/trail handle it).
            # Allowed if in loss (cut losses) OR past 50% TP (profit already secured).
            allow_voluntary_close = False
            # Voluntary close requires a fresh model decision — only on new bars
            if new_bar and action == 3 and not force_close:
                half_tp = tp_pct * 0.5
                if upnl_pct < 0:
                    allow_voluntary_close = True
                    close_reason = "VOLUNTARY"
                elif upnl_pct >= half_tp:
                    allow_voluntary_close = True
                    close_reason = "VOLUNTARY"
                    log.info(f"{tag} 💰 Voluntary close allowed — profit {upnl_pct*100:+.2f}% past 50% TP")
                else:
                    log.info(f"{tag} 🔒 Voluntary close blocked — profit {upnl_pct*100:+.2f}% < 50% TP, let trail handle exit")

            if force_close or allow_voluntary_close:
                pos = state['position']
                qty = state['qty']
                if pos == 1:
                    order = close_long(exec_client, symbol, qty)
                else:
                    order = close_short(exec_client, symbol, qty)

                if order:
                    # SAFETY FIX: confirm exit fill before updating state
                    filled, exit_avg_price, exit_exec_qty = confirm_order_fill(
                        exec_client, symbol, order, requested_qty=qty,
                        tag=f"{tag} EXIT")

                    if not filled:
                        # SAFETY FIX: retry close once if not confirmed
                        log.critical(f"{tag} EXIT order NOT confirmed — retrying close")
                        if pos == 1:
                            order2 = close_long(exec_client, symbol, qty)
                        else:
                            order2 = close_short(exec_client, symbol, qty)
                        if order2:
                            filled, exit_avg_price, exit_exec_qty = confirm_order_fill(
                                exec_client, symbol, order2, requested_qty=qty,
                                tag=f"{tag} EXIT_RETRY")
                        if not filled:
                            log.critical(f"{tag} EXIT STILL NOT CONFIRMED after retry "
                                         f"— MANUAL INTERVENTION REQUIRED — state NOT updated")
                            return  # SAFETY FIX: do not update state if exit unconfirmed

                    # SAFETY FIX: use actual fill price for PnL, not current_price
                    exit_price = exit_avg_price if exit_avg_price > 0 else current_price
                    if exit_avg_price <= 0:
                        log.warning(f"{tag} ⚠️  Exit avgPrice=0 — using current_price "
                                    f"${current_price:,.2f} as fallback")

                    pnl_pct = (exit_price - entry) / entry * pos  # SAFETY FIX: exit_price not current_price
                    pnl_usdt = pnl_pct * TRADE_NOTIONAL
                    state['total_pnl_usdt'] += pnl_usdt
                    if pnl_pct > 0:
                        state['wins'] += 1
                        state['consecutive_losses'] = 0
                    else:
                        state['losses'] += 1
                        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1
                        if state['consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
                            log.critical(f"{tag} {symbol} HIT {MAX_CONSECUTIVE_LOSSES} CONSECUTIVE LOSSES — DISABLING SYMBOL")
                            disabled_symbols.add(symbol)
                            save_disabled_symbols(disabled_symbols)  # SAFETY FIX: persist
                    wr = state['wins'] / max(1, state['wins'] + state['losses'])
                    side_name = "LONG" if pos == 1 else "SHORT"
                    log.info(f"{tag} 💰 {side_name} closed @ ${exit_price:,.2f} | "
                             f"PnL: ${pnl_usdt:+.2f} ({pnl_pct*100:+.2f}%) | "
                             f"WR: {wr:.1%} | Total: ${state['total_pnl_usdt']:+.2f}")
                    log_trade_event(
                        symbol, "CLOSE", side_name, qty, exit_price,  # SAFETY FIX: exit_price
                        trade_id=state.get('last_trade_id', ''),
                        pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                        close_reason=close_reason or "UNKNOWN",
                        confidence=state.get('entry_confidence', 0.0),
                        verdict=state.get('entry_verdict', ''))

                    if state.get('entry_conditions'):
                        try:
                            memory.record_trade(
                                conditions=state['entry_conditions'],
                                action="BUY" if pos == 1 else "SELL",
                                entry_price=entry,
                                exit_price=exit_price,  # SAFETY FIX: actual fill price
                                pnl_pct=pnl_pct,
                                pnl_usdt=pnl_usdt,
                                confidence=state.get('entry_confidence', 0.5),
                                verdict=state.get('entry_verdict', 'UNKNOWN'),
                                reasoning=state.get('entry_reasoning', ''),
                                trade_type="live"
                            )
                        except Exception as e:
                            log.error(f"{tag} ❌ Memory record failed: {e}")

                    state['position'] = 0
                    state['entry_price'] = 0.0
                    state['entry_time'] = None
                    state['qty'] = 0.0
                    state['entry_conditions'] = None
                    state['entry_confidence'] = 0.5
                    state['entry_verdict'] = None
                    state['entry_reasoning'] = ""
                    state['breakeven_set'] = False
                    state['trail_1r_set'] = False
                    state['sl_price'] = 0.0  # SL_SOFTWARE: reset SL on close
                    save_state(state, state_path)
                else:
                    # SAFETY FIX: close order placement itself failed — log critical
                    log.critical(f"{tag} EXIT order placement FAILED — position still open, "
                                 f"exchange SL should protect")

        # Per-bar summary (skipped on intra-bar SL-monitoring cycles)
        if new_bar:
            wr = state['wins'] / max(1, state['wins'] + state['losses'])
            sentiment_agent = ctx.get('sentiment_agent')
            if sentiment_agent:
                try:
                    sig = sentiment_agent.get_signal(symbol)
                    log.info(
                        f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | "
                        f"PnL: ${state['total_pnl_usdt']:+.2f} | "
                        f"Sentiment: {sig['direction']} ({sig['strength']:.0%}) | "
                        f"F&G: {sig['fear_greed_label']}({sig['fear_greed_value']})"
                    )
                except Exception:
                    log.info(f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")
            else:
                log.info(f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")

    except Exception as e:
        if is_rate_limit_error(e):
            note_binance_ban(e)   # RATE_LIMIT: record ban, main loop goes silent
        else:
            log.error(f"{tag} ❌ Cycle error: {e}", exc_info=True)


# ── Agent initialization ──────────────────────────────────────────────────────

def init_agents(exec_client):
    """Initialize the multi-agent system."""
    sentiment_agent  = SentimentAgent(binance_client=exec_client)
    position_auditor = PositionAuditor()
    log.info("✅ SentimentAgent initialized (CryptoCompare + CoinGecko + Yahoo + F&G + Binance)")
    log.info("✅ PositionAuditor initialized")
    return sentiment_agent, position_auditor


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    if not api_key or not api_secret:
        log.error("❌ Set BINANCE_API_KEY and BINANCE_SECRET_KEY env vars")
        return

    log.info("🚀 Crypto Live Agent starting (FUTURES TESTNET, MULTI-SYMBOL)")
    log.info(f"   Symbols:   {SYMBOLS}")
    log.info(f"   Notional:  ${TRADE_NOTIONAL} USDT per trade per symbol")
    log.info(f"   Leverage:  {LEVERAGE}×")
    log.info(f"   Model:     {MODEL_PATH}")

    data_client = Client(api_key, api_secret, requests_params={"timeout": 20})
    data_client.ping = lambda: None

    exec_client = Client(api_key, api_secret, requests_params={"timeout": 20})
    exec_client.ping = lambda: None
    exec_client.FUTURES_URL = FUTURES_TESTNET_URL

    try:
        bal = exec_client.futures_account_balance()
        usdt = next((float(x['balance']) for x in bal if x['asset'] == 'USDT'), 0.0)
        log.info(f"✅ Connected to futures testnet | USDT balance: {usdt:,.2f}")
    except Exception as e:
        log.error(f"❌ Connection failed: {e}")
        return

    # SAFETY FIX: capture starting balance from exchange — used for kill switch
    starting_balance = fetch_live_balance(exec_client)
    if starting_balance is None or starting_balance <= 0:
        log.error(f"❌ Cannot fetch starting balance ({starting_balance}) — aborting")
        return
    max_loss_per_trade = TRADE_NOTIONAL * 0.03
    log.info(f"💰 Starting balance: ${starting_balance:,.2f} | "
             f"Max loss/trade: ${max_loss_per_trade:.2f} | "
             f"Kill switch: {KILL_SWITCH_DRAWDOWN_PCT:.0%} drawdown = "
             f"${starting_balance * KILL_SWITCH_DRAWDOWN_PCT:,.2f}")

    # Margin sanity — scaled by number of symbols (worst case: all simultaneously in position)
    required_margin_per_trade = TRADE_NOTIONAL / LEVERAGE
    total_margin_worst_case = required_margin_per_trade * len(SYMBOLS)
    if starting_balance < total_margin_worst_case * 1.5:
        log.warning(f"⚠️  Low balance: ${starting_balance:.2f} may be insufficient for worst-case "
                    f"${total_margin_worst_case:.2f} margin (${required_margin_per_trade:.2f} × {len(SYMBOLS)} symbols)")

    # Per-symbol leverage setup
    for symbol in SYMBOLS:
        try:
            exec_client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
            log.info(f"✅ Leverage set to {LEVERAGE}× for {symbol}")
        except Exception as e:
            log.warning(f"⚠️  Could not set leverage for {symbol}: {e}")

    # Load model (shared across all symbols — trained on BTC, generalizes via raw OHLCV)
    if not os.path.exists(MODEL_PATH):
        log.error(f"❌ Model not found: {MODEL_PATH}")
        return
    model = PPO.load(MODEL_PATH)
    log.info(f"✅ Model loaded: {MODEL_PATH}")

    sentiment_agent, position_auditor = init_agents(exec_client)

    # Per-symbol memories, states, and filters
    memories = {}
    states = {}
    filters = {}
    last_bar_times = {}
    for symbol in SYMBOLS:
        mem_path = memory_path_for(symbol)
        log_csv  = trade_log_path_for(symbol)
        if not os.path.exists(mem_path):
            log.error(f"❌ Memory file missing for {symbol}: {mem_path} — run historical_trainer.py + regret_trainer.py first")
            return
        memories[symbol] = TradeMemory(memory_file=mem_path, log_file=log_csv)
        log.info(f"✅ {symbol} memory loaded: {memories[symbol].memory['total_trades']} historical trades")

        states[symbol] = load_state_with_migration(symbol)
        log.info(f"✅ {symbol} state: position={states[symbol]['position']}, "
                 f"trades={states[symbol]['trade_count']}, "
                 f"PnL=${states[symbol]['total_pnl_usdt']:+.2f}")

        filters[symbol] = get_futures_symbol_info(exec_client, symbol)
        log.info(f"✅ {symbol} filters: tick={filters[symbol]['tick_size']}, "
                 f"step={filters[symbol]['step_size']}, "
                 f"min_qty={filters[symbol]['min_qty']}")

        last_bar_times[symbol] = None

    log.info(f"\n{'='*60}")
    log.info(f"📡 Multi-symbol live loop active — {len(SYMBOLS)} symbols, every 5 min")
    log.info(f"{'='*60}\n")

    # SAFETY FIX: load disabled symbols from disk — persists across restarts
    disabled_symbols = load_disabled_symbols()
    if disabled_symbols:
        log.critical(f"🚫 Symbols disabled from previous session: {disabled_symbols} "
                     f"— delete {DISABLED_SYMBOLS_PATH} to re-enable")

    ctx = {
        'data_client': data_client,
        'exec_client': exec_client,
        'model': model,
        'memories': memories,
        'states': states,
        'filters': filters,
        'last_bar_times': last_bar_times,
        'max_loss_per_trade': max_loss_per_trade,
        'disabled_symbols': disabled_symbols,
        'sentiment_agent':  sentiment_agent,
        'position_auditor': position_auditor,
    }

    while True:
        try:
            # ── RATE_LIMIT: during an IP ban, make ZERO data-API requests. ──
            # Every request made while banned EXTENDS the ban. Sleep it out.
            if binance_banned_now():
                log_ban_throttled()
                time.sleep(min(300, max(30, ban_remaining_secs())))
                continue

            # SAFETY FIX: exchange-based kill switch — fresh balance every cycle
            current_balance = fetch_live_balance(exec_client)
            if current_balance is None:
                log.error("🛑 Cannot fetch balance — skipping cycle for safety")
                time.sleep(FETCH_EVERY)
                continue
            drawdown = current_balance - starting_balance
            if drawdown < -(starting_balance * KILL_SWITCH_DRAWDOWN_PCT):
                log.error(f"🛑 GLOBAL KILL SWITCH: balance ${current_balance:,.2f} = "
                          f"${drawdown:+,.2f} from start ${starting_balance:,.2f} "
                          f"(>{KILL_SWITCH_DRAWDOWN_PCT:.0%} drawdown). "
                          f"Shutting down ALL symbols.")
                return

            # ── Per-symbol cycle (try/except inside process_symbol — one bad symbol can't stall others) ──
            for symbol in SYMBOLS:
                process_symbol(symbol, ctx)

            if disabled_symbols:
                log.critical(f"🚫 Disabled symbols: {disabled_symbols}")
            active_count = len(SYMBOLS) - len(disabled_symbols)
            log.info(f"\n💼 Cycle done | {active_count}/{len(SYMBOLS)} active | "
                     f"Balance: ${current_balance:,.2f} | Drawdown: ${drawdown:+,.2f}\n")

        except KeyboardInterrupt:
            log.info("\n⏹️  Agent stopped by user")
            break
        except Exception as e:
            log.error(f"❌ Loop error: {e}", exc_info=True)

        time.sleep(FETCH_EVERY)


if __name__ == "__main__":
    main()
