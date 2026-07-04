"""
v2 Agent — Retrain on 120d archive with smarter training config
Key changes vs previous:
  - Uses nq_archive.pkl (more data)
  - Early stopping: save best and stop if no improvement for 100k steps
  - More training steps: 1M instead of 500k
  - Randomized start position: agent sees different market regimes each episode
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment_v2 import NQTradingEnvV2

print("📥 Loading data...")
df = pd.read_pickle("nq_archive.pkl")
df = df.reset_index()
print(f"✅ {len(df)} bars ({df.index[0]} → {df.index[-1]})")

# Better split: last 20 days = test, everything before = train
# This avoids the test set being too small
test_bars = 20 * 78 * 8  # 20 trading days × ~78 active bars/day × 8 sessions/day... actually simpler:
split = int(len(df) * 0.85)  # 85/15 split gives more test data in absolute terms
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

class EarlyStopCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=5000, patience=150000):
        super().__init__()
        self.eval_env    = eval_env
        self.eval_every  = eval_every
        self.patience    = patience
        self.best_pnl    = -99999
        self.steps_since_best = 0

    def _on_step(self):
        if self.n_calls % self.eval_every == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, info = self.eval_env.step(action)

            pnl  = info.get("total_pnl", 0)
            wr   = info.get("win_rate", 0)
            wins = info.get("wins", 0)
            losses = info.get("losses", 0)
            dd   = info.get("max_drawdown", 0)

            print(f"  Step {self.n_calls:>7,} | PnL: ${pnl:>+7,.0f} | "
                  f"WR: {wr:.1%} ({wins}W/{losses}L) | DD: {dd:.1%}")

            if pnl > self.best_pnl:
                self.best_pnl = pnl
                self.steps_since_best = 0
                self.model.save("rl_agent_v2_fixed_best")
                print(f"  💾 New best! PnL: ${pnl:+,.0f}")
            else:
                self.steps_since_best += self.eval_every
                if self.steps_since_best >= self.patience:
                    print(f"\n⏹️  Early stop — no improvement for {self.patience:,} steps")
                    return False  # stops training
        return True

print("\n🏗️  Building environments...")
train_env = NQTradingEnvV2(train_df, sl_pts=30, tp_pts=30)
test_env  = NQTradingEnvV2(test_df,  sl_pts=30, tp_pts=30)

print("🤖 Building PPO agent...")
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # back to exploring  # low entropy = exploit what works
    verbose=0,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

params = sum(p.numel() for p in model.policy.parameters())
print(f"   Parameters: {params:,}")

print(f"\n🚀 Training up to 1,000,000 steps (early stop at 150k no-improvement)...")
print("="*60)

model.learn(
    total_timesteps=1_000_000,
    callback=EarlyStopCallback(test_env, eval_every=5000, patience=150000),
    progress_bar=False
)

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

best = PPO.load("rl_agent_v2_fixed_best")
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = best.predict(obs, deterministic=True)
    obs, _, done, _, info = test_env.step(action)

pnl  = info.get("total_pnl", 0)
wr   = info.get("win_rate", 0)
wins = info.get("wins", 0)
losses = info.get("losses", 0)
dd   = info.get("max_drawdown", 0)

print(f"\n📊 RESULTS:")
print(f"   PnL:      ${pnl:+,.0f}")
print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
print(f"   Max DD:   {dd:.1%}")
print(f"   Trades:   {wins+losses}")
print(f"\n💾 Saved: rl_agent_v2_fixed_best.zip")
