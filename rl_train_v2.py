"""
Phase 3: Train PPO Agent on ICT-Aware Environment v2
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment_v2 import NQTradingEnvV2

print("📥 Loading 60d NQ data...")
df = pd.read_pickle("nq_60d.pkl")
df = df.reset_index()
print(f"✅ {len(df)} bars loaded")

split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df)} bars | Test: {len(test_df)} bars")

class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=10000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.best_pnl = -99999

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
                self.model.save("rl_agent_v2_best")
                print(f"  💾 New best! PnL: ${pnl:+,.0f}")
        return True

print("\n🏗️  Building v2 environments (ICT-aware)...")
train_env = NQTradingEnvV2(train_df, sl_pts=30, tp_pts=30)
test_env  = NQTradingEnvV2(test_df,  sl_pts=30, tp_pts=30)

print("🤖 Building PPO agent (28 features)...")
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    verbose=0,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

params = sum(p.numel() for p in model.policy.parameters())
print(f"   Network: 28 → 256 → 256 → 128 → 3")
print(f"   Parameters: {params:,}")

TOTAL_STEPS = 500_000
print(f"\n🚀 Training {TOTAL_STEPS:,} steps...")
print("="*60)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=TrainingCallback(test_env, eval_every=10000),
    progress_bar=False
)

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

best = PPO.load("rl_agent_v2_best")
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

print(f"\n📊 v2 AGENT RESULTS:")
print(f"   PnL:      ${pnl:+,.0f}")
print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
print(f"   Max DD:   {dd:.1%}")
print(f"   Trades:   {wins+losses}")
print(f"\n📊 v1 AGENT (baseline):")
print(f"   PnL:      +$2,280")
print(f"   Win Rate: 60.9%")
print(f"   Trades:   174")

if wr > 0.609:
    print(f"\n✅ v2 beats v1 — ICT features improved the agent!")
elif pnl > 2280:
    print(f"\n✅ v2 more profitable than v1!")
elif pnl > 0:
    print(f"\n⚠️  v2 profitable but below v1 — needs more training")
else:
    print(f"\n❌ v2 still losing — ICT gates too restrictive for this data")

print(f"\n💾 Saved: rl_agent_v2_best.zip")
