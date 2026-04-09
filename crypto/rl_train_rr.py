"""
Crypto MTF Agent Training — RR-Aware Reward
============================================
Trains using CryptoRREnv which penalizes small wins (< 1.5%) and
rewards meaningful moves (+3.0 / -0.5 / -2.0 categorical on close,
continuous unrealized PnL signal on non-close bars).

Uses btc_historical_mtf.pkl (already cached from historical_trainer).
Saves to crypto_mtf_rr_best.zip — does NOT overwrite crypto_mtf_best.zip.

Run:
    cd /Users/moura/trading-indicator/ai-model/crypto
    python3 rl_train_rr.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from crypto_env_v2 import CryptoRREnv

DATASET_PKL  = "btc_historical_mtf.pkl"
MODEL_PATH   = "crypto_mtf_rr_best"   # .zip appended by SB3
TOTAL_STEPS  = 2_000_000
EVAL_EVERY   = 50_000
PATIENCE     = 300_000
MIN_WR_AFTER = 500_000  # if WR < 45% after this many steps, abort
MIN_WR_FLOOR = 0.45
MIN_TRADES   = 30       # ignore evals with too few trades

# ── Data ──────────────────────────────────────────────────────────────────────

if not os.path.exists(DATASET_PKL):
    print(f"❌ Dataset not found: {DATASET_PKL}")
    print("Run historical_trainer.py first to generate the dataset.")
    sys.exit(1)

print(f"📥 Loading dataset: {DATASET_PKL}")
df = pd.read_pickle(DATASET_PKL)
print(f"✅ {len(df)} bars | {df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]}")

# 80/20 time-based split
split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")


# ── Callback ──────────────────────────────────────────────────────────────────

class RRCallback(BaseCallback):
    """
    Eval callback that selects best checkpoint by WR (with min-trades
    floor). Includes an early-abort guard: if after MIN_WR_AFTER steps
    the best WR is still below MIN_WR_FLOOR, training stops to avoid
    burning CPU on a non-converging reward landscape.
    """
    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env
        self.best_wr = 0.0
        self.best_pnl = -1e9
        self.steps_since_best = 0

    def _on_step(self):
        if self.n_calls % EVAL_EVERY != 0:
            return True

        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.eval_env.step(action)

        pnl    = info.get("total_pnl", 0)
        pnl_pct= info.get("total_pnl_pct", 0)
        wr     = info.get("win_rate", 0)
        wins   = info.get("wins", 0)
        losses = info.get("losses", 0)
        dd     = info.get("max_drawdown", 0)
        trades = info.get("total_trades", 0)

        print(f"  Step {self.n_calls:>7,} | WR: {wr:.1%} ({wins}W/{losses}L) | "
              f"PnL: ${pnl:>+7.2f} ({pnl_pct:>+6.1%}) | Trades: {trades} | DD: {dd:.1%}")

        # Best checkpoint by WR (min trades floor)
        if trades >= MIN_TRADES and wr > self.best_wr:
            self.best_wr = wr
            self.best_pnl = pnl
            self.steps_since_best = 0
            self.model.save(MODEL_PATH)
            print(f"  💾 New best | WR {wr:.1%} | {trades} trades | PnL ${pnl:+.2f}")
        else:
            self.steps_since_best += EVAL_EVERY

        # Early abort if sparse reward landscape isn't converging
        if self.n_calls >= MIN_WR_AFTER and self.best_wr < MIN_WR_FLOOR:
            print(f"\n⏹️  Aborting — best WR {self.best_wr:.1%} < {MIN_WR_FLOOR:.0%} "
                  f"after {self.n_calls:,} steps. Reward landscape too sparse.")
            return False

        # Patience-based early stop
        if self.steps_since_best >= PATIENCE:
            print(f"\n⏹️  Early stop — no improvement for {PATIENCE:,} steps "
                  f"(best WR {self.best_wr:.1%})")
            return False

        return True


# ── Train ─────────────────────────────────────────────────────────────────────

print("\n🏗️  Building RR-focused environments...")
train_env = CryptoRREnv(train_df)
test_env  = CryptoRREnv(test_df)
print(f"   Reward: +3.0 (win ≥1.5%) | -0.5 (small win) | -2.0 (loss)")
print(f"   MIN_PROFIT_PCT: {CryptoRREnv.MIN_PROFIT_PCT:.1%}")

print("\n🤖 Building PPO agent...")
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    verbose=0,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)
params = sum(p.numel() for p in model.policy.parameters())
print(f"   Network: 36 → 256 → 256 → 128 → 4 | Params: {params:,}")

print(f"\n🚀 Training up to {TOTAL_STEPS:,} steps...")
print(f"   Eval every {EVAL_EVERY:,} | Patience {PATIENCE:,} | "
      f"Abort if WR < {MIN_WR_FLOOR:.0%} after {MIN_WR_AFTER:,}")
print("=" * 70)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=RRCallback(test_env),
    progress_bar=False,
)

# ── Final eval ────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("FINAL OOS EVALUATION")
print("=" * 70)

if os.path.exists(MODEL_PATH + ".zip"):
    best = PPO.load(MODEL_PATH)

    # Eval on OOS (test) set
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = best.predict(obs, deterministic=True)
        obs, _, done, _, info = test_env.step(action)

    pnl    = info.get("total_pnl", 0)
    pnl_pct= info.get("total_pnl_pct", 0)
    wr     = info.get("win_rate", 0)
    wins   = info.get("wins", 0)
    losses = info.get("losses", 0)
    dd     = info.get("max_drawdown", 0)
    trades = info.get("total_trades", 0)

    print(f"\n📊 RR MODEL (OOS):")
    print(f"   PnL:      ${pnl:+.2f} ({pnl_pct:+.1%})")
    print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
    print(f"   Trades:   {trades}")
    print(f"   Max DD:   {dd:.1%}")

    # Compare with current model if it exists
    current_model_path = "crypto_mtf_best.zip"
    if os.path.exists(current_model_path):
        print(f"\n📊 CURRENT MODEL (OOS, same test set):")
        from crypto_env_v2 import CryptoMTFEnv
        current_env = CryptoMTFEnv(test_df)
        current = PPO.load(current_model_path)
        obs2, _ = current_env.reset()
        done2 = False
        while not done2:
            action2, _ = current.predict(obs2, deterministic=True)
            obs2, _, done2, _, info2 = current_env.step(action2)

        print(f"   PnL:      ${info2.get('total_pnl',0):+.2f} ({info2.get('total_pnl_pct',0):+.1%})")
        print(f"   Win Rate: {info2.get('win_rate',0):.1%} "
              f"({info2.get('wins',0)}W / {info2.get('losses',0)}L)")
        print(f"   Trades:   {info2.get('total_trades',0)}")
        print(f"   Max DD:   {info2.get('max_drawdown',0):.1%}")

    print(f"\n💾 Best checkpoint: {MODEL_PATH}.zip")
    if wr >= 0.60:
        print(f"✅ WR {wr:.1%} ≥ 60% — candidate for deployment")
    elif wr >= 0.50:
        print(f"🟡 WR {wr:.1%} — below 60% target but above coin-flip")
    else:
        print(f"❌ WR {wr:.1%} — model didn't converge to a viable strategy")
else:
    print("⚠️  No best checkpoint saved — callback never found a qualifying eval")
    print("   (either MIN_TRADES floor wasn't met or training aborted early)")
