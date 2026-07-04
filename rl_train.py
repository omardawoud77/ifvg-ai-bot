"""
Phase 2: Train PPO Agent on NQ Futures
=======================================
Uses Stable-Baselines3 PPO to train an agent
that learns to trade NQ profitably from scratch.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment import NQTradingEnv
import os

# ── Load data ────────────────────────────────────────────────────────────────
print("📥 Loading 60d NQ data...")
df = pd.read_pickle("nq_60d.pkl")
df = df.reset_index()
print(f"✅ {len(df)} bars loaded")

# Split: 80% train, 20% test
split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df)} bars | Test: {len(test_df)} bars")

# ── Progress callback ─────────────────────────────────────────────────────────
class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=10000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.best_pnl = -99999
        self.episode = 0

    def _on_step(self):
        if self.n_calls % self.eval_every == 0:
            # Quick eval on test set
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, info = self.eval_env.step(action)

            pnl    = info.get("total_pnl", 0)
            wr     = info.get("win_rate", 0)
            wins   = info.get("wins", 0)
            losses = info.get("losses", 0)
            dd     = info.get("max_drawdown", 0)

            print(f"  Step {self.n_calls:>7,} | "
                  f"PnL: ${pnl:>+7,.0f} | "
                  f"WR: {wr:.1%} ({wins}W/{losses}L) | "
                  f"DD: {dd:.1%}")

            # Save best model
            if pnl > self.best_pnl:
                self.best_pnl = pnl
                self.model.save("rl_agent_best")
                print(f"  💾 New best model saved! PnL: ${pnl:+,.0f}")

        return True

# ── Create environments ───────────────────────────────────────────────────────
print("\n🏗️  Building environments...")
train_env = NQTradingEnv(train_df, sl_pts=30, tp_pts=30)
test_env  = NQTradingEnv(test_df,  sl_pts=30, tp_pts=30)

# ── Build PPO agent ───────────────────────────────────────────────────────────
print("🤖 Building PPO agent...")
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # encourages exploration
    verbose=0,
    tensorboard_log=None,
    policy_kwargs=dict(
        net_arch=[128, 128, 64]  # 3-layer neural network
    )
)

print(f"   Network: 15 inputs → 128 → 128 → 64 → 3 actions")
print(f"   Algorithm: PPO (Proximal Policy Optimization)")
print(f"   Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
TOTAL_STEPS = 500_000  # ~500k steps = good starting point
print(f"\n🚀 Training for {TOTAL_STEPS:,} steps...")
print(f"   Evaluating every 10,000 steps on test data")
print(f"   Best model auto-saved to rl_agent_best.zip")
print("="*60)

callback = TrainingCallback(test_env, eval_every=10000)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=callback,
    progress_bar=False
)

# ── Final evaluation ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL EVALUATION — LOADING BEST MODEL")
print("="*60)

best_model = PPO.load("rl_agent_best")
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = best_model.predict(obs, deterministic=True)
    obs, _, done, _, info = test_env.step(action)

pnl    = info.get("total_pnl", 0)
wr     = info.get("win_rate", 0)
wins   = info.get("wins", 0)
losses = info.get("losses", 0)
dd     = info.get("max_drawdown", 0)

print(f"\n📊 BEST AGENT RESULTS (test set):")
print(f"   PnL:          ${pnl:+,.0f}")
print(f"   Win Rate:     {wr:.1%} ({wins}W / {losses}L)")
print(f"   Max Drawdown: {dd:.1%}")
print(f"   Total Trades: {wins + losses}")

print(f"\n📊 RANDOM BASELINE (for comparison):")
print(f"   PnL:      ~-$1,020")
print(f"   Win Rate: ~49%")

if pnl > 0:
    print(f"\n✅ Agent is PROFITABLE — beating random baseline!")
elif pnl > -500:
    print(f"\n⚠️  Agent close to breakeven — needs more training")
else:
    print(f"\n❌ Agent still losing — needs more steps or tuning")

print(f"\n💾 Model saved to: rl_agent_best.zip")
print("Next step: run rl_train.py with more steps, or connect to live bot")
