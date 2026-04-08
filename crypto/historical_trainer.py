"""
Historical Trainer
==================
Feeds 2020-2025 BTC data through the reasoning engine
to build trade_memory.json before live deployment.

Run once:
    cd /Users/moura/trading-indicator/ai-model/crypto
    python3 historical_trainer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from binance.client import Client
from stable_baselines3 import PPO
from crypto_env_v2 import CryptoMTFEnv
from reasoning_engine import perceive, interpret, decide
from trade_memory import TradeMemory

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL      = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
SL_PCT      = 0.02       # 2% stop loss
TP_PCT      = 0.04       # 4% take profit (2R)
MAX_BARS    = 48         # max hold time in bars
MODEL_PATH  = "crypto_mtf_best.zip"
# Per-symbol output paths — BTCUSDT keeps the legacy filenames for backward compat
MEMORY_FILE = "trade_memory.json" if SYMBOL == "BTCUSDT" else f"trade_memory_{SYMBOL.lower()}.json"
LOG_FILE    = "trade_log.csv" if SYMBOL == "BTCUSDT" else f"trade_log_{SYMBOL.lower()}.csv"
DATASET_PKL = "btc_historical_mtf.pkl" if SYMBOL == "BTCUSDT" else f"{SYMBOL.lower()}_historical_mtf.pkl"
HIST_LOG    = "historical_trade_log.csv" if SYMBOL == "BTCUSDT" else f"historical_trade_log_{SYMBOL.lower()}.csv"

# ── Fetch historical data ─────────────────────────────────────────────────────

def fetch_historical_mtf(symbol="BTCUSDT"):
    print(f"📥 Fetching historical {symbol} data 2020-2025...")

    client = Client("", "")
    client.ping = lambda: None

    def fetch_tf(interval, start, limit=1000):
        all_klines = []
        while True:
            klines = client.get_historical_klines(
                symbol, interval, start, limit=limit
            )
            if not klines:
                break
            all_klines.extend(klines)
            last_ts = klines[-1][0]
            start = last_ts + 1
            if len(klines) < limit:
                break
            print(f"  Fetched {len(all_klines)} bars...", end="\r")
        return all_klines

    def to_df(klines):
        df = pd.DataFrame(klines, columns=[
            'ts','open','high','low','close','volume',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df[['ts','open','high','low','close','volume']].set_index('ts')

    start_date = "1 Jan, 2020"

    print("  Fetching 1H bars...")
    df_1h = to_df(fetch_tf(Client.KLINE_INTERVAL_1HOUR, start_date))
    print(f"  ✅ 1H: {len(df_1h)} bars")

    print("  Fetching 4H bars...")
    df_4h = to_df(fetch_tf(Client.KLINE_INTERVAL_4HOUR, start_date))
    print(f"  ✅ 4H: {len(df_4h)} bars")

    print("  Fetching 1D bars...")
    df_1d = to_df(fetch_tf(Client.KLINE_INTERVAL_1DAY, start_date))
    print(f"  ✅ 1D: {len(df_1d)} bars")

    print("  Fetching 1W bars...")
    df_1w = to_df(fetch_tf(Client.KLINE_INTERVAL_1WEEK, start_date))
    print(f"  ✅ 1W: {len(df_1w)} bars")

    df_4h_ff = df_4h.reindex(df_1h.index, method='ffill').add_prefix('h4_')
    df_1d_ff = df_1d.reindex(df_1h.index, method='ffill').add_prefix('d1_')
    df_1w_ff = df_1w.reindex(df_1h.index, method='ffill').add_prefix('w1_')

    df = pd.concat([df_1h, df_4h_ff, df_1d_ff, df_1w_ff], axis=1).dropna()
    df = df.reset_index().rename(columns={'ts': 'Datetime'})

    print(f"\n✅ MTF dataset: {len(df)} bars ({df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]})")
    return df


# ── Simulate a single trade ───────────────────────────────────────────────────

def simulate_trade(df, entry_idx, direction, sl_pct=SL_PCT, tp_pct=TP_PCT, max_bars=MAX_BARS):
    entry_price = float(df.iloc[entry_idx]['close'])

    if direction == 1:  # long
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:  # short
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    for i in range(1, max_bars + 1):
        if entry_idx + i >= len(df):
            break

        bar = df.iloc[entry_idx + i]
        high = float(bar.get('high', bar['close']))
        low = float(bar.get('low', bar['close']))
        close = float(bar['close'])

        if direction == 1:
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                return pnl, i, "SL"
            if high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                return pnl, i, "TP"
        else:
            if high >= sl_price:
                pnl = (entry_price - sl_price) / entry_price
                return pnl, i, "SL"
            if low <= tp_price:
                pnl = (entry_price - tp_price) / entry_price
                return pnl, i, "TP"

    # Time exit
    exit_price = float(df.iloc[min(entry_idx + max_bars, len(df)-1)]['close'])
    pnl = (exit_price - entry_price) / entry_price * direction
    return pnl, max_bars, "TIME"


# ── Main training loop ────────────────────────────────────────────────────────

def train(df, model, memory):
    print(f"\n🧠 Starting historical training on {len(df)} bars...")
    print("This will take 10-20 minutes. Progress shown every 1000 bars.\n")

    trades_recorded = 0
    bars_processed = 0
    skipped_off_hours = 0

    # We need at least 50 bars of history before we start
    start_idx = 50

    for i in range(start_idx, len(df) - MAX_BARS - 1):
        bars_processed += 1

        if bars_processed % 1000 == 0:
            wr = memory.memory['total_wins'] / max(1, memory.memory['total_trades'])
            print(f"  Bar {i}/{len(df)} | Trades: {trades_recorded} | WR: {wr:.1%}")

        # Build observation
        window = df.iloc[:i+1].reset_index(drop=True)

        try:
            env = CryptoMTFEnv(window)
            env.reset()
            obs = env._get_obs(len(window) - 1)
        except Exception:
            continue

        # Inject flat position state
        obs[27] = 0.0
        obs[28] = 0.0
        obs[29] = 0.0
        obs[30] = 0.0

        # Get PPO action
        try:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        except Exception:
            continue

        # Only process entry signals
        if action not in (1, 2):
            continue

        # Build perception and interpret
        fake_state = {"position": 0, "entry_price": 0, "entry_time": None, "qty": 0}

        try:
            perception = perceive(window, fake_state)
            conditions, narrative = interpret(perception)
        except Exception:
            continue

        # Skip off-hours (matches live agent behavior)
        if conditions['session'] == 'OFF_HOURS':
            skipped_off_hours += 1
            continue

        # Get reasoning verdict
        try:
            verdict, confidence, reasoning_text = decide(
                action, conditions, perception, memory, narrative
            )
        except Exception:
            continue

        # Only simulate trades the reasoning engine would execute
        if verdict not in ("EXECUTE", "WEAK_EXECUTE"):
            continue

        # Simulate the trade outcome
        direction = 1 if action == 1 else -1
        try:
            pnl_pct, bars_held, exit_reason = simulate_trade(df, i, direction)
        except Exception:
            continue

        entry_price = float(df.iloc[i]['close'])
        exit_price = entry_price * (1 + pnl_pct * direction) if direction == 1 \
            else entry_price * (1 - pnl_pct)

        # Record in memory
        memory.record_trade(
            conditions=conditions,
            action="BUY" if action == 1 else "SELL",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_usdt=pnl_pct * 110,
            confidence=confidence,
            verdict=verdict,
            reasoning=reasoning_text[:200],
            trade_type="historical"
        )

        trades_recorded += 1

        # Skip ahead to avoid overlapping trades
        i_next = i + bars_held

    return trades_recorded


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("✅ Loading PPO model...")
    model = PPO.load(MODEL_PATH)

    # Initialize fresh memory
    if os.path.exists(MEMORY_FILE):
        os.rename(MEMORY_FILE, MEMORY_FILE + ".backup")
        print(f"⚠️  Backed up existing memory to {MEMORY_FILE}.backup")

    memory = TradeMemory(memory_file=MEMORY_FILE, log_file=HIST_LOG)

    # Fetch data
    df = fetch_historical_mtf(SYMBOL)

    # Save the dataset for reuse
    df.to_pickle(DATASET_PKL)
    print(f"💾 Saved dataset to {DATASET_PKL}")

    # Train
    total_trades = train(df, model, memory)

    # Update metadata
    memory.memory["meta"]["training_bars"] = len(df)
    memory.save()

    # Print summary
    print(f"\n✅ Training complete — {total_trades} trades recorded")
    memory.print_summary()

    print("\n📁 Files generated:")
    print(f"  — {MEMORY_FILE}  (agent's brain — commit this to GitHub)")
    print(f"  — {HIST_LOG}  (full trade history)")
    print(f"  — {DATASET_PKL}  (dataset cache)")
    print(f"\n🚀 Ready to deploy. Commit {MEMORY_FILE} and push to GitHub.")
