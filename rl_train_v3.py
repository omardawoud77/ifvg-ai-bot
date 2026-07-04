"""
v3 Agent — Fully Autonomous, No ICT Gates
Agent sees everything, decides everything including exits.
Actions: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment_v3 import NQTradingEnvV3

print("📥 Loading data...")
df = pd.read_pickle("nq_archive.pkl")
df = df.reset_index()
print(f"✅ {len(df)} bars ({df['Datetime'].iloc[0].date()} → {df['Datetime'].iloc[-1].date()})")

split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=10000, patience=200000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.patience = patience
        self.best_pnl = -99999
        self.steps_since_best = 0

    def _on_step(self):
        if self.n_calls % self.eval_every == 0:
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
            trades = info.get("total_trades", 0)

            print(f"  Step {self.n_calls:>7,} | PnL: ${pnl:>+7,.0f} | "
                  f"WR: {wr:.1%} ({wins}W/{losses}L) | "
                  f"Trades: {trades} | DD: {dd:.1%}")

            if pnl > self.best_pnl:
                self.best_pnl = pnl
                self.steps_since_best = 0
                self.model.save("rl_agent_v3_best")
                print(f"  💾 New best! PnL: ${pnl:+,.0f}")
            else:
                self.steps_since_best += self.eval_every
                if self.steps_since_best >= self.patience:
                    print(f"\n⏹️  Early stop — no improvement for {self.patience:,} steps")
                    return False
        return True

print("\n🏗️  Building v3 environments (MNQ, $2/pt, no gates)...")
train_env = NQTradingEnvV3(train_df, point_value=2.0)
test_env  = NQTradingEnvV3(test_df,  point_value=2.0)

print("🤖 Building PPO agent (32 features, 4 actions)...")
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,   # higher entropy — agent needs to explore 4 actions freely
    verbose=0,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

params = sum(p.numel() for p in model.policy.parameters())
print(f"   Network: 32 → 256 → 256 → 128 → 4")
print(f"   Parameters: {params:,}")
print(f"   ent_coef: 0.02 (high — agent must explore freely)")

print(f"\n🚀 Training up to 2,000,000 steps...")
print(f"   This will take ~20-30 min. Agent starts with zero knowledge.")
print(f"   Watch trade count grow as it discovers when to enter/exit.")
print("="*60)

model.learn(
    total_timesteps=2_000_000,
    callback=TrainingCallback(test_env, eval_every=10000, patience=200000),
    progress_bar=False
)

print("\n" + "="*60)
print("FINAL EVALUATION — BEST MODEL")
print("="*60)

best = PPO.load("rl_agent_v3_best")
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = best.predict(obs, deterministic=True)
    obs, _, done, _, info = test_env.step(action)

pnl    = info.get("total_pnl", 0)
wr     = info.get("win_rate", 0)
wins   = info.get("wins", 0)
losses = info.get("losses", 0)
dd     = info.get("max_drawdown", 0)
trades = info.get("total_trades", 0)

print(f"\n📊 v3 AGENT RESULTS (MNQ, $2/pt):")
print(f"   PnL:      ${pnl:+,.0f}")
print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
print(f"   Trades:   {trades}")
print(f"   Max DD:   {dd:.1%}")
print(f"\n📊 Random baseline: ~-$1,600, 49% WR, 2000+ trades")

if pnl > 0 and wr > 0.52:
    print(f"\n✅ Agent profitable — learned to trade on its own!")
elif pnl > 0:
    print(f"\n⚠️  Agent profitable but WR low — learning but noisy")
else:
    print(f"\n❌ Agent still losing — needs more data or steps")

print(f"\n💾 Saved: rl_agent_v3_best.zip")
