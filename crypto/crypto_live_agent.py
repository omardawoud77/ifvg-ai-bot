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
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
from stable_baselines3 import PPO
from crypto_env_v2 import CryptoMTFEnv
from reasoning_engine import perceive, interpret, decide, get_dynamic_sl_tp, simulate_missed_trade
from trade_memory import TradeMemory

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOLS         = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRADE_NOTIONAL  = 1500.0         # USDT notional per trade per symbol
LEVERAGE        = 3              # 3× leverage
FETCH_EVERY     = 300            # seconds between cycles (5 min)
MAX_DAILY_TRADES = 6             # max NEW entries per UTC day PER SYMBOL (18 total across 3)
MODEL_PATH      = "crypto_mtf_best.zip"
LOG_PATH        = "crypto_live_agent.log"
LEGACY_BTC_STATE = "crypto_live_state.json"  # pre-multisymbol path, migrated on first load
MIN_BARS        = 50

FUTURES_TESTNET_URL = "https://testnet.binancefuture.com/fapi"

HERE = os.path.dirname(os.path.abspath(__file__))

def state_path_for(symbol):
    return os.path.join(HERE, f"state_{symbol.lower()}.json")

def memory_path_for(symbol):
    """BTC keeps the legacy filename for backward compat with the calibrated regret memory."""
    if symbol == "BTCUSDT":
        return os.path.join(HERE, "trade_memory.json")
    return os.path.join(HERE, f"trade_memory_{symbol.lower()}.json")

def trade_log_path_for(symbol):
    if symbol == "BTCUSDT":
        return os.path.join(HERE, "trade_log.csv")
    return os.path.join(HERE, f"trade_log_{symbol.lower()}.csv")

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
    "sl_pct": 0.02,
    "tp_pct": 0.04,
    "entry_conditions": None,
    "entry_confidence": 0.5,
    "entry_verdict": None,
    "entry_reasoning": "",
    "last_rejected_action": None,
    "last_rejected_conditions": None,
    "last_rejected_bar_ts": None,
    "last_rejected_sl_pct": 0.02,
    "last_rejected_tp_pct": 0.04,
    "breakeven_set": False,
    "trail_1r_set": False,
}

def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            loaded = json.load(f)
        for k, v in DEFAULT_STATE.items():
            loaded.setdefault(k, v)
        return loaded
    return dict(DEFAULT_STATE)

def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

def load_state_with_migration(symbol):
    """For BTCUSDT, migrate from the legacy single-symbol state file if needed."""
    new_path = state_path_for(symbol)
    if os.path.exists(new_path):
        return load_state(new_path)

    if symbol == "BTCUSDT":
        legacy = os.path.join(HERE, LEGACY_BTC_STATE)
        if os.path.exists(legacy):
            log.info(f"📦 Migrating BTC state from {LEGACY_BTC_STATE} → state_btcusdt.json")
            with open(legacy) as f:
                loaded = json.load(f)
            for k, v in DEFAULT_STATE.items():
                loaded.setdefault(k, v)
            save_state(loaded, new_path)
            # Leave legacy file in place for rollback safety
            return loaded

    return dict(DEFAULT_STATE)

def fetch_live_balance(client):
    try:
        bal = client.futures_account_balance()
        usdt = next((float(x['balance']) for x in bal if x['asset'] == 'USDT'), 0.0)
        return usdt
    except Exception as e:
        log.error(f"❌ Could not fetch balance: {e}")
        return 1000.0

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

    df_1h = fetch_tf(Client.KLINE_INTERVAL_1HOUR,  bars)
    df_4h = fetch_tf(Client.KLINE_INTERVAL_4HOUR,  bars)
    df_1d = fetch_tf(Client.KLINE_INTERVAL_1DAY,   bars)
    df_1w = fetch_tf(Client.KLINE_INTERVAL_1WEEK,  bars)

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


# ── Per-symbol cycle logic ────────────────────────────────────────────────────

def process_symbol(symbol, ctx):
    """
    Run one bar-cycle of logic for a single symbol. Mutates ctx['states'][symbol]
    and ctx['last_bar_times'][symbol] in place. Catches all exceptions so one
    bad symbol can't stall the others.
    """
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
        df = fetch_mtf(data_client, symbol, bars=200)
        if len(df) < MIN_BARS:
            log.warning(f"{tag} Not enough bars ({len(df)} < {MIN_BARS})")
            return

        latest_bar_time = df['Datetime'].iloc[-2]
        if latest_bar_time == ctx['last_bar_times'][symbol]:
            return  # no new bar for this symbol

        ctx['last_bar_times'][symbol] = latest_bar_time
        current_price = float(df['close'].iloc[-2])
        log.info(f"\n{tag} 📊 New bar: {latest_bar_time} | Close: ${current_price:,.2f}")

        # ── Regret review ────────────────────────────────────────────────────
        if (state.get('last_rejected_action') is not None and state.get('position') == 0):
            try:
                rejected_ts = state.get('last_rejected_bar_ts')
                rejected_action = state['last_rejected_action']
                rejected_conditions = state['last_rejected_conditions']
                r_sl = state.get('last_rejected_sl_pct', 0.02)
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
                pnl_pct = (current_price - state['entry_price']) / state['entry_price'] * state['position']
                pnl_usdt = pnl_pct * TRADE_NOTIONAL
                state['total_pnl_usdt'] += pnl_usdt
                if pnl_pct > 0:
                    state['wins'] += 1
                else:
                    state['losses'] += 1
                if state.get('entry_conditions'):
                    try:
                        memory.record_trade(
                            conditions=state['entry_conditions'],
                            action="BUY" if state['position'] == 1 else "SELL",
                            entry_price=state['entry_price'],
                            exit_price=current_price,
                            pnl_pct=pnl_pct,
                            pnl_usdt=pnl_usdt,
                            confidence=state.get('entry_confidence', 0.5),
                            verdict=state.get('entry_verdict', 'UNKNOWN'),
                            reasoning=state.get('entry_reasoning', '') + " [resynced]",
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
                save_state(state, state_path)
            elif exchange_qty != 0 and state['position'] == 0:
                log.warning(f"{tag} ⚠️  Exchange has qty={exchange_qty} but state=flat — manual resolution needed")
                log.warning(f"{tag}    Skipping this bar to avoid double-trading")
                return
        except Exception as e:
            log.error(f"{tag} ❌ Position reconcile failed: {e}")

        # ── Build observation ────────────────────────────────────────────────
        env = CryptoMTFEnv(df.iloc[:-1].reset_index(drop=True))
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

        # ── Model decision ───────────────────────────────────────────────────
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_names = {0: "HOLD", 1: "BUY (long)", 2: "SELL (short)", 3: "CLOSE"}
        log.info(f"{tag} 🤖 Decision: {action_names[action]} | Position: {state['position']}")

        # ── Reasoning gate ───────────────────────────────────────────────────
        perception = perceive(df, state)
        conditions, narrative = interpret(perception)
        verdict, confidence, reasoning_text = decide(action, conditions, perception, memory, narrative)
        log.info(f"\n{'='*55}\n{tag} {reasoning_text}\n{'='*55}")

        if action in (1, 2) and verdict not in ("EXECUTE", "WEAK_EXECUTE"):
            sl_pct_rej, tp_pct_rej = get_dynamic_sl_tp(conditions, memory)
            state['last_rejected_action'] = action
            state['last_rejected_conditions'] = conditions
            state['last_rejected_bar_ts'] = latest_bar_time.isoformat() if hasattr(latest_bar_time, 'isoformat') else str(latest_bar_time)
            state['last_rejected_sl_pct'] = sl_pct_rej
            state['last_rejected_tp_pct'] = tp_pct_rej
            save_state(state, state_path)
            log.info(f"{tag} ⛔ Trade REJECTED — stored for regret review next bar")
            wr = state['wins'] / max(1, state['wins'] + state['losses'])
            log.info(f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")
            return

        # ── Daily trade limit gate (per symbol) ──────────────────────────────
        today = datetime.now(timezone.utc).date().isoformat()
        if state.get('last_trade_date') != today:
            state['daily_trades'] = 0
            state['last_trade_date'] = today
        entries_allowed = state['daily_trades'] < MAX_DAILY_TRADES
        if not entries_allowed and state['position'] == 0 and action in (1, 2):
            log.warning(f"{tag} ⏸️  Daily trade limit reached ({state['daily_trades']}/{MAX_DAILY_TRADES}) — skip entry")

        # ── Execute ──────────────────────────────────────────────────────────
        if state['position'] == 0 and entries_allowed:
            if action in (1, 2):
                sl_pct, tp_pct = get_dynamic_sl_tp(conditions, memory)
                log.info(f"{tag} 📐 Dynamic SL: {sl_pct:.1%} | TP: {tp_pct:.1%} | WR basis: {memory.get_win_rate(conditions)}")

            if action == 1:  # LONG
                order, qty = open_long(exec_client, symbol, TRADE_NOTIONAL, current_price,
                                       step_size, min_qty, qty_prec)
                if order:
                    state['position'] = 1
                    state['entry_price'] = current_price
                    state['entry_time'] = datetime.now(timezone.utc).isoformat()
                    state['qty'] = qty
                    state['trade_count'] += 1
                    state['daily_trades'] += 1
                    state['sl_pct'] = sl_pct
                    state['tp_pct'] = tp_pct
                    state['entry_conditions'] = conditions
                    state['entry_confidence'] = confidence
                    state['entry_verdict'] = verdict
                    state['entry_reasoning'] = reasoning_text[:300]
                    save_state(state, state_path)
                    log.info(f"{tag} 📈 LONG opened | qty={qty} @ ${current_price:,.2f}")
                    try:
                        sl_price = round_price(current_price * (1 - sl_pct), tick_size, price_prec)
                        exec_client.futures_create_order(
                            symbol=symbol,
                            side="SELL",
                            type="STOP_MARKET",
                            stopPrice=sl_price,
                            closePosition="true",
                        )
                        log.info(f"{tag} 🛡️  Exchange SL placed @ ${sl_price}")
                    except Exception as e:
                        log.error(f"{tag} ❌ SL placement failed: {e}")

            elif action == 2:  # SHORT
                order, qty = open_short(exec_client, symbol, TRADE_NOTIONAL, current_price,
                                        step_size, min_qty, qty_prec)
                if order:
                    state['position'] = -1
                    state['entry_price'] = current_price
                    state['entry_time'] = datetime.now(timezone.utc).isoformat()
                    state['qty'] = qty
                    state['trade_count'] += 1
                    state['daily_trades'] += 1
                    state['sl_pct'] = sl_pct
                    state['tp_pct'] = tp_pct
                    state['entry_conditions'] = conditions
                    state['entry_confidence'] = confidence
                    state['entry_verdict'] = verdict
                    state['entry_reasoning'] = reasoning_text[:300]
                    save_state(state, state_path)
                    log.info(f"{tag} 📉 SHORT opened | qty={qty} @ ${current_price:,.2f}")
                    try:
                        sl_price = round_price(current_price * (1 + sl_pct), tick_size, price_prec)
                        exec_client.futures_create_order(
                            symbol=symbol,
                            side="BUY",
                            type="STOP_MARKET",
                            stopPrice=sl_price,
                            closePosition="true",
                        )
                        log.info(f"{tag} 🛡️  Exchange SL placed @ ${sl_price}")
                    except Exception as e:
                        log.error(f"{tag} ❌ SL placement failed: {e}")

        elif state['position'] != 0:  # in a position
            sl_pct = state.get('sl_pct', 0.02)
            tp_pct = state.get('tp_pct', 0.04)
            entry = state['entry_price']
            force_close = False
            close_reason = None

            # Unrealized PnL (used for account protection + voluntary close gate)
            upnl_pct = (current_price - entry) / entry * state['position']

            # SL / TP checks
            if state['position'] == 1:
                if current_price <= entry * (1 - sl_pct):
                    log.info(f"{tag} 🛑 LONG SL @ ${current_price:,.2f}")
                    force_close = True
                    close_reason = "SL"
                elif current_price >= entry * (1 + tp_pct):
                    log.info(f"{tag} 🎯 LONG TP @ ${current_price:,.2f}")
                    force_close = True
                    close_reason = "TP"
            elif state['position'] == -1:
                if current_price >= entry * (1 + sl_pct):
                    log.info(f"{tag} 🛑 SHORT SL @ ${current_price:,.2f}")
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
                        be_price = round_price(entry, tick_size, price_prec)
                        exec_client.futures_create_order(
                            symbol=symbol,
                            side=be_side,
                            type="STOP_MARKET",
                            stopPrice=be_price,
                            closePosition="true",
                        )
                        log.info(f"{tag} 🛡️  Exchange SL replaced @ ${be_price} (breakeven)")
                    except Exception as e:
                        log.error(f"{tag} ❌ Breakeven SL placement failed: {e}")

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
                        exec_client.futures_create_order(
                            symbol=symbol,
                            side=trail_side,
                            type="STOP_MARKET",
                            stopPrice=trail_sl_price,
                            closePosition="true",
                        )
                        log.info(f"{tag} 🛡️  Exchange SL replaced @ ${trail_sl_price} (1R lock)")
                    except Exception as e:
                        log.error(f"{tag} ❌ Trail SL placement failed: {e}")

            # ── Voluntary close gate ──
            # Blocked if profit < 50% TP (let breakeven/trail handle it).
            # Allowed if in loss (cut losses) OR past 50% TP (profit already secured).
            allow_voluntary_close = False
            if action == 3 and not force_close:
                half_tp = tp_pct * 0.5
                if upnl_pct < 0:
                    allow_voluntary_close = True
                elif upnl_pct >= half_tp:
                    allow_voluntary_close = True
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
                    pnl_pct = (current_price - entry) / entry * pos
                    pnl_usdt = pnl_pct * TRADE_NOTIONAL
                    state['total_pnl_usdt'] += pnl_usdt
                    if pnl_pct > 0:
                        state['wins'] += 1
                    else:
                        state['losses'] += 1
                    wr = state['wins'] / max(1, state['wins'] + state['losses'])
                    side_name = "LONG" if pos == 1 else "SHORT"
                    log.info(f"{tag} 💰 {side_name} closed @ ${current_price:,.2f} | "
                             f"PnL: ${pnl_usdt:+.2f} ({pnl_pct*100:+.2f}%) | "
                             f"WR: {wr:.1%} | Total: ${state['total_pnl_usdt']:+.2f}")

                    if state.get('entry_conditions'):
                        try:
                            memory.record_trade(
                                conditions=state['entry_conditions'],
                                action="BUY" if pos == 1 else "SELL",
                                entry_price=entry,
                                exit_price=current_price,
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
                    save_state(state, state_path)

        wr = state['wins'] / max(1, state['wins'] + state['losses'])
        log.info(f"{tag} 📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")

    except Exception as e:
        log.error(f"{tag} ❌ Cycle error: {e}", exc_info=True)


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

    # Live balance + per-trade loss cap + global kill switch
    live_balance = fetch_live_balance(exec_client)
    max_loss_per_trade = TRADE_NOTIONAL * 0.03
    daily_loss_limit = -(live_balance * 0.10)
    last_balance_refresh = datetime.now(timezone.utc)
    log.info(f"💰 Balance: ${live_balance:,.2f} | Max loss/trade: ${max_loss_per_trade:.2f} | Global kill switch: ${daily_loss_limit:.2f}")

    # Margin sanity — scaled by number of symbols (worst case: all simultaneously in position)
    required_margin_per_trade = TRADE_NOTIONAL / LEVERAGE
    total_margin_worst_case = required_margin_per_trade * len(SYMBOLS)
    if live_balance < total_margin_worst_case * 1.5:
        log.warning(f"⚠️  Low balance: ${live_balance:.2f} may be insufficient for worst-case "
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

    ctx = {
        'data_client': data_client,
        'exec_client': exec_client,
        'model': model,
        'memories': memories,
        'states': states,
        'filters': filters,
        'last_bar_times': last_bar_times,
        'max_loss_per_trade': max_loss_per_trade,
    }

    while True:
        try:
            # Refresh live balance every 24h
            hours_since_refresh = (datetime.now(timezone.utc) - last_balance_refresh).total_seconds() / 3600
            if hours_since_refresh >= 24:
                live_balance = fetch_live_balance(exec_client)
                ctx['max_loss_per_trade'] = TRADE_NOTIONAL * 0.03
                daily_loss_limit = -(live_balance * 0.10)
                last_balance_refresh = datetime.now(timezone.utc)
                log.info(f"💰 Balance refreshed: ${live_balance:,.2f} | Kill switch: ${daily_loss_limit:.2f}")

            # ── Global kill switch — sum PnL across all symbols ──
            total_pnl_all = sum(states[s]['total_pnl_usdt'] for s in SYMBOLS)
            if total_pnl_all < daily_loss_limit:
                log.error(f"🛑 GLOBAL kill switch: total PnL ${total_pnl_all:.2f} < ${daily_loss_limit:.2f}. Shutting down all symbols.")
                return

            # ── Per-symbol cycle (try/except inside process_symbol — one bad symbol can't stall others) ──
            for symbol in SYMBOLS:
                process_symbol(symbol, ctx)

            log.info(f"\n💼 Cycle done | Total PnL across {len(SYMBOLS)} symbols: ${total_pnl_all:+.2f}\n")

        except KeyboardInterrupt:
            log.info("\n⏹️  Agent stopped by user")
            break
        except Exception as e:
            log.error(f"❌ Loop error: {e}", exc_info=True)

        time.sleep(FETCH_EVERY)


if __name__ == "__main__":
    main()
