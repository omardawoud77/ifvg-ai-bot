"""
Regret Trainer
==============
Walks 5 years of BTC history, simulates every rejection,
records regret scores so the agent launches already calibrated.

Run after historical_trainer.py:
    python3 regret_trainer.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from crypto_env_v2 import CryptoMTFEnv
from reasoning_engine import perceive, interpret, decide, get_dynamic_sl_tp, simulate_missed_trade
from trade_memory import TradeMemory

SYMBOL     = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
MODEL_PATH = "crypto_mtf_best.zip"
MEMORY_FILE = f"trade_memory_{SYMBOL.lower()}.json" if SYMBOL != "BTCUSDT" else "trade_memory.json"
DATASET    = f"{SYMBOL.lower()}_historical_mtf.pkl" if SYMBOL != "BTCUSDT" else "btc_historical_mtf.pkl"
MAX_BARS   = 48

def run_regret_training(df, model, memory):
    print(f"\n🔁 Starting regret training on {len(df)} bars...")
    print("Walking history, simulating every rejected trade...\n")

    rejections_reviewed = 0
    would_have_won = 0
    would_have_lost = 0
    bars_processed = 0
    start_idx = 50

    i = start_idx
    while i < len(df) - MAX_BARS - 1:
        bars_processed += 1

        if bars_processed % 2000 == 0:
            wr = would_have_won / max(1, rejections_reviewed)
            print(f"  Bar {i}/{len(df)} | Rejections reviewed: {rejections_reviewed} | Missed WR: {wr:.1%}")

        window = df.iloc[:i+1].reset_index(drop=True)

        try:
            env = CryptoMTFEnv(window)
            env.reset()
            obs = env._get_obs(len(window) - 1)
            obs[27] = 0.0
            obs[28] = 0.0
            obs[29] = 0.0
            obs[30] = 0.0
        except Exception:
            i += 1
            continue

        try:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        except Exception:
            i += 1
            continue

        if action not in (1, 2):
            i += 1
            continue

        fake_state = {"position": 0, "entry_price": 0, "entry_time": None, "qty": 0}

        try:
            perception = perceive(window, fake_state)
            conditions, narrative = interpret(perception)
        except Exception:
            i += 1
            continue

        if conditions['session'] == 'OFF_HOURS':
            i += 1
            continue

        try:
            verdict, confidence, reasoning_text = decide(
                action, conditions, perception, memory, narrative
            )
        except Exception:
            i += 1
            continue

        if verdict in ("EXECUTE", "WEAK_EXECUTE"):
            # Trade was taken — skip, already in memory from historical_trainer
            i += 1
            continue

        # Trade was REJECTED — simulate what would have happened
        try:
            sl_pct, tp_pct = get_dynamic_sl_tp(conditions, memory)
            was_profitable, pnl, exit_reason = simulate_missed_trade(
                df, i, action, sl_pct, tp_pct, MAX_BARS
            )
            if was_profitable is not None:
                memory.record_missed_trade(conditions, was_profitable)
                rejections_reviewed += 1
                if was_profitable:
                    would_have_won += 1
                else:
                    would_have_lost += 1
        except Exception:
            pass

        i += 1

    return rejections_reviewed, would_have_won, would_have_lost


if __name__ == "__main__":
    if not os.path.exists(DATASET):
        print(f"❌ Dataset not found: {DATASET}")
        print("Run historical_trainer.py first to generate the dataset.")
        sys.exit(1)

    if not os.path.exists(MEMORY_FILE):
        print(f"❌ Memory file not found: {MEMORY_FILE}")
        print("Run historical_trainer.py first.")
        sys.exit(1)

    print(f"✅ Loading dataset: {DATASET}")
    df = pd.read_pickle(DATASET)
    print(f"   {len(df)} bars loaded")

    print(f"✅ Loading memory: {MEMORY_FILE}")
    memory = TradeMemory(memory_file=MEMORY_FILE, log_file="regret_training.log")
    print(f"   {memory.memory['total_trades']} existing trades in memory")

    print(f"✅ Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    reviewed, won, lost = run_regret_training(df, model, memory)

    print(f"\n✅ Regret training complete")
    print(f"   Rejections reviewed: {reviewed}")
    print(f"   Would have WON: {won} ({won/max(1,reviewed):.1%})")
    print(f"   Would have LOST: {lost} ({lost/max(1,reviewed):.1%})")
    print(f"\n📊 Top regret setups (agent was wrong to reject):")

    regret_summary = memory.get_regret_summary(min_missed=5)
    for r in regret_summary[:15]:
        print(f"   {r['condition']}: regret={r['regret_score']:.1%} ({r['missed_profitable']}/{r['missed_total']} profitable misses)")

    print(f"\n💾 Memory updated: {MEMORY_FILE}")
    print(f"🚀 Agent is now regret-calibrated. Deploy when ready.")
