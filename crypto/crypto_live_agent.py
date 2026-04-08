"""
Crypto Live Trading Agent — Binance Futures Testnet
====================================================
Runs the trained MTF agent live on Binance USDT-M perpetual futures testnet.
Fetches 1h spot bars (mainnet, public) for the model, executes trades on
the futures testnet (no real money).

Usage:
  export BINANCE_API_KEY=your_key
  export BINANCE_SECRET_KEY=your_secret
  python3 crypto_live_agent.py

Both LONG and SHORT supported. 1× leverage (no margin risk).
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

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL          = "BTCUSDT"
TRADE_NOTIONAL  = 110.0          # USDT notional per trade
LEVERAGE        = 1              # 1× — no margin risk
FETCH_EVERY     = 300            # seconds between bar checks (5 min)
SL_PCT          = 0.02           # 2% hard stop
DAILY_LOSS_LIMIT = -20.0         # USD — stop trading if total loss exceeds this
MAX_DAILY_TRADES = 6             # max NEW entries per UTC day
MODEL_PATH      = "crypto_mtf_best.zip"
LOG_PATH        = "crypto_live_agent.log"
STATE_PATH      = "crypto_live_state.json"
MIN_BARS        = 50

FUTURES_TESTNET_URL = "https://testnet.binancefuture.com/fapi"

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
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
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
    }

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

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
    lot_filter = next(f for f in sym['filters'] if f['filterType'] == 'LOT_SIZE')
    step_size = float(lot_filter['stepSize'])
    min_qty   = float(lot_filter['minQty'])
    qty_precision = sym.get('quantityPrecision', 3)
    return step_size, min_qty, qty_precision

def round_qty(qty, step_size, precision):
    rounded = float(int(qty / step_size) * step_size)
    return round(rounded, precision)

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
        log.info(f"✅ Futures order: {side} {qty} {symbol} (reduceOnly={reduce_only})")
        return order
    except BinanceAPIException as e:
        log.error(f"❌ Order failed: {e}")
        return None

def open_long(client, symbol, notional, price, step, min_qty, prec):
    qty = round_qty(notional / price, step, prec)
    if qty < min_qty:
        log.warning(f"Qty {qty} < min {min_qty} — skip")
        return None, 0.0
    order = place_futures_order(client, symbol, "BUY", qty, reduce_only=False)
    return order, qty

def open_short(client, symbol, notional, price, step, min_qty, prec):
    qty = round_qty(notional / price, step, prec)
    if qty < min_qty:
        log.warning(f"Qty {qty} < min {min_qty} — skip")
        return None, 0.0
    order = place_futures_order(client, symbol, "SELL", qty, reduce_only=False)
    return order, qty

def close_long(client, symbol, qty):
    return place_futures_order(client, symbol, "SELL", qty, reduce_only=True)

def close_short(client, symbol, qty):
    return place_futures_order(client, symbol, "BUY", qty, reduce_only=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")

    if not api_key or not api_secret:
        log.error("❌ Set BINANCE_API_KEY and BINANCE_SECRET_KEY env vars")
        return

    log.info("🚀 Crypto Live Agent starting (FUTURES TESTNET)")
    log.info(f"   Symbol:    {SYMBOL}")
    log.info(f"   Notional:  ${TRADE_NOTIONAL} USDT")
    log.info(f"   Leverage:  {LEVERAGE}× (no margin risk)")
    log.info(f"   Hard SL:   {SL_PCT*100}%")
    log.info(f"   Model:     {MODEL_PATH}")

    # Single client — spot mainnet for data, futures testnet for trades
    client = Client(api_key, api_secret, requests_params={"timeout": 20})
    client.ping = lambda: None  # skip geo-blocked spot ping
    client.FUTURES_URL = FUTURES_TESTNET_URL

    # Verify connection
    try:
        bal = client.futures_account_balance()
        usdt = next((float(x['balance']) for x in bal if x['asset'] == 'USDT'), 0.0)
        log.info(f"✅ Connected to futures testnet | USDT balance: {usdt:,.2f}")
    except Exception as e:
        log.error(f"❌ Connection failed: {e}")
        return

    # Set leverage to 1×
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        log.info(f"✅ Leverage set to {LEVERAGE}× for {SYMBOL}")
    except Exception as e:
        log.warning(f"⚠️  Could not set leverage: {e}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        log.error(f"❌ Model not found: {MODEL_PATH}")
        return
    model = PPO.load(MODEL_PATH)
    log.info(f"✅ Model loaded: {MODEL_PATH}")

    # Symbol filters
    step_size, min_qty, qty_prec = get_futures_symbol_info(client, SYMBOL)
    log.info(f"✅ Symbol filters: step={step_size}, min_qty={min_qty}, precision={qty_prec}")

    # State
    state = load_state()
    log.info(f"✅ State: position={state['position']}, trades={state['trade_count']}, PnL=${state['total_pnl_usdt']:+.2f}")

    last_bar_time = None
    log.info(f"\n{'='*60}")
    log.info("📡 Live loop active — checking every 5 min for new 1h bar")
    log.info(f"{'='*60}\n")

    while True:
        try:
            # Daily loss limit kill switch
            if state['total_pnl_usdt'] < DAILY_LOSS_LIMIT:
                log.error(f"🛑 Loss limit hit: ${state['total_pnl_usdt']:.2f} < ${DAILY_LOSS_LIMIT:.2f}. Shutting down.")
                return

            df = fetch_mtf(client, SYMBOL, bars=200)
            if len(df) < MIN_BARS:
                log.warning(f"Not enough bars ({len(df)} < {MIN_BARS})")
                time.sleep(FETCH_EVERY)
                continue

            latest_bar_time = df['Datetime'].iloc[-2]
            if latest_bar_time == last_bar_time:
                time.sleep(FETCH_EVERY)
                continue

            last_bar_time = latest_bar_time
            current_price = float(df['close'].iloc[-2])
            log.info(f"\n📊 New bar: {latest_bar_time} | Close: ${current_price:,.0f}")

            # Reconcile local state with exchange position
            try:
                pos_info = client.futures_position_information(symbol=SYMBOL)
                exchange_qty = float(pos_info[0]['positionAmt'])
                if exchange_qty == 0 and state['position'] != 0:
                    log.warning(f"⚠️  Exchange flat but state={state['position']} — SL likely fired, resyncing")
                    # Approximate PnL (use current price as exit since we don't know exact stop fill)
                    pnl_pct = (current_price - state['entry_price']) / state['entry_price'] * state['position']
                    pnl_usdt = pnl_pct * TRADE_NOTIONAL
                    state['total_pnl_usdt'] += pnl_usdt
                    if pnl_pct > 0:
                        state['wins'] += 1
                    else:
                        state['losses'] += 1
                    state['position'] = 0
                    state['entry_price'] = 0.0
                    state['entry_time'] = None
                    state['qty'] = 0.0
                    save_state(state)
                elif exchange_qty != 0 and state['position'] == 0:
                    log.warning(f"⚠️  Exchange has qty={exchange_qty} but state=flat — manual resolution needed")
                    log.warning("   Skipping this bar to avoid double-trading")
                    time.sleep(FETCH_EVERY)
                    continue
            except Exception as e:
                log.error(f"❌ Position reconcile failed: {e}")

            # Build observation
            env = CryptoMTFEnv(df.iloc[:-1].reset_index(drop=True))
            env.reset()
            obs = env._get_obs(len(df) - 2)

            # Inject real position state
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

            # Model decision
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_names = {0: "HOLD", 1: "BUY (long)", 2: "SELL (short)", 3: "CLOSE"}
            log.info(f"🤖 Decision: {action_names[action]} | Position: {state['position']}")

            # ── Daily trade limit gate (entries only — closes always allowed) ─
            today = datetime.now(timezone.utc).date().isoformat()
            if state.get('last_trade_date') != today:
                state['daily_trades'] = 0
                state['last_trade_date'] = today
            entries_allowed = state['daily_trades'] < MAX_DAILY_TRADES
            if not entries_allowed and state['position'] == 0 and action in (1, 2):
                log.warning(f"⏸️  Daily trade limit reached ({state['daily_trades']}/{MAX_DAILY_TRADES}) — skip entry")

            # ── Execute ────────────────────────────────────────────────
            if state['position'] == 0 and entries_allowed:
                if action == 1:  # open LONG
                    order, qty = open_long(client, SYMBOL, TRADE_NOTIONAL, current_price,
                                           step_size, min_qty, qty_prec)
                    if order:
                        state['position'] = 1
                        state['entry_price'] = current_price
                        state['entry_time'] = datetime.now(timezone.utc).isoformat()
                        state['qty'] = qty
                        state['trade_count'] += 1
                        state['daily_trades'] += 1
                        save_state(state)  # save IMMEDIATELY after fill
                        log.info(f"📈 LONG opened | qty={qty} @ ${current_price:,.0f}")
                        # Place exchange stop-market — instant SL, no polling required
                        try:
                            sl_price = round(current_price * (1 - SL_PCT), 1)
                            client.futures_create_order(
                                symbol=SYMBOL,
                                side="SELL",
                                type="STOP_MARKET",
                                stopPrice=sl_price,
                                closePosition="true",
                            )
                            log.info(f"🛡️  Exchange SL placed @ ${sl_price:,.1f}")
                        except Exception as e:
                            log.error(f"❌ SL placement failed: {e}")

                elif action == 2:  # open SHORT
                    order, qty = open_short(client, SYMBOL, TRADE_NOTIONAL, current_price,
                                            step_size, min_qty, qty_prec)
                    if order:
                        state['position'] = -1
                        state['entry_price'] = current_price
                        state['entry_time'] = datetime.now(timezone.utc).isoformat()
                        state['qty'] = qty
                        state['trade_count'] += 1
                        state['daily_trades'] += 1
                        save_state(state)
                        log.info(f"📉 SHORT opened | qty={qty} @ ${current_price:,.0f}")
                        # Place exchange stop-market — instant SL, no polling required
                        try:
                            sl_price = round(current_price * (1 + SL_PCT), 1)
                            client.futures_create_order(
                                symbol=SYMBOL,
                                side="BUY",
                                type="STOP_MARKET",
                                stopPrice=sl_price,
                                closePosition="true",
                            )
                            log.info(f"🛡️  Exchange SL placed @ ${sl_price:,.1f}")
                        except Exception as e:
                            log.error(f"❌ SL placement failed: {e}")

            else:  # in a position
                # Hard SL check
                force_close = False
                if state['position'] == 1 and current_price <= state['entry_price'] * (1 - SL_PCT):
                    log.info(f"🛑 LONG SL triggered @ ${current_price:,.0f}")
                    force_close = True
                elif state['position'] == -1 and current_price >= state['entry_price'] * (1 + SL_PCT):
                    log.info(f"🛑 SHORT SL triggered @ ${current_price:,.0f}")
                    force_close = True

                if action == 3 or force_close:
                    pos = state['position']
                    qty = state['qty']
                    if pos == 1:
                        order = close_long(client, SYMBOL, qty)
                    else:
                        order = close_short(client, SYMBOL, qty)

                    if order:
                        pnl_pct = (current_price - state['entry_price']) / state['entry_price'] * pos
                        pnl_usdt = pnl_pct * TRADE_NOTIONAL
                        state['total_pnl_usdt'] += pnl_usdt
                        if pnl_pct > 0:
                            state['wins'] += 1
                        else:
                            state['losses'] += 1
                        wr = state['wins'] / max(1, state['wins'] + state['losses'])
                        side_name = "LONG" if pos == 1 else "SHORT"
                        log.info(f"💰 {side_name} closed @ ${current_price:,.0f} | "
                                 f"PnL: ${pnl_usdt:+.2f} ({pnl_pct*100:+.2f}%) | "
                                 f"WR: {wr:.1%} | Total: ${state['total_pnl_usdt']:+.2f}")
                        state['position'] = 0
                        state['entry_price'] = 0.0
                        state['entry_time'] = None
                        state['qty'] = 0.0
                        save_state(state)

            # Periodic status
            wr = state['wins'] / max(1, state['wins'] + state['losses'])
            log.info(f"📋 Trades: {state['trade_count']} | WR: {wr:.1%} | PnL: ${state['total_pnl_usdt']:+.2f}")

        except KeyboardInterrupt:
            log.info("\n⏹️  Agent stopped by user")
            break
        except Exception as e:
            log.error(f"❌ Loop error: {e}", exc_info=True)

        time.sleep(FETCH_EVERY)

if __name__ == "__main__":
    main()
