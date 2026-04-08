"""
Crypto PPO Agent Training — BTC/USDT Spot
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from binance.client import Client
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from crypto_env import CryptoTradingEnv

# ── Fetch data ────────────────────────────────────────────────────────────────
print("📥 Fetching BTC/USDT 5m data from Binance...")
client = Client("", "", tld='com')
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "60 days ago UTC")

df = pd.DataFrame(klines, columns=[
    'Datetime','open','high','low','close','volume',
    'close_time','qav','trades','tbbav','tbqav','ignore'
])
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
for col in ['open','high','low','close','volume']:
    df[col] = df[col].astype(float)
df = df[['Datetime','open','high','low','close','volume']].reset_index(drop=True)
print(f"✅ {len(df)} bars ({df['Datetime'].iloc[0].date()} → {df['Datetime'].iloc[-1].date()})")

split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df):,} | Test: {len(test_df):,}")

class TrainingCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=10000, patience=150000):
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
            pnl_pct= info.get("total_pnl_pct", 0)
            wr     = info.get("win_rate", 0)
            wins   = info.get("wins", 0)
            losses = info.get("losses", 0)
            dd     = info.get("max_drawdown", 0)
            trades = info.get("total_trades", 0)

            print(f"  Step {self.n_calls:>7,} | PnL: ${pnl:>+7.2f} ({pnl_pct:>+6.1%}) | "
                  f"WR: {wr:.1%} ({wins}W/{losses}L) | Trades: {trades} | DD: {dd:.1%}")

            if pnl > self.best_pnl:
                self.best_pnl = pnl
                self.steps_since_best = 0
                self.model.save("crypto_agent_best")
                print(f"  💾 New best! PnL: ${pnl:+.2f}")
            else:
                self.steps_since_best += self.eval_every
                if self.steps_since_best >= self.patience:
                    print(f"\n⏹️  Early stop")
                    return False
        return True

print("\n🏗️  Building environments...")
train_env = CryptoTradingEnv(train_df)
test_env  = CryptoTradingEnv(test_df)

print("🤖 Building PPO agent (27 features, spot BTC)...")
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
    ent_coef=0.02,
    verbose=0,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)
params = sum(p.numel() for p in model.policy.parameters())
print(f"   Network: 27 → 256 → 256 → 128 → 4 | Params: {params:,}")

print(f"\n🚀 Training up to 1,000,000 steps...")
print("="*65)

model.learn(
    total_timesteps=1_000_000,
    callback=TrainingCallback(test_env, eval_every=10000, patience=150000),
    progress_bar=False
)

print("\n" + "="*65)
print("FINAL EVALUATION")
print("="*65)

best = PPO.load("crypto_agent_best")
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

print(f"\n📊 CRYPTO AGENT RESULTS (BTC/USDT spot, $1,000 capital):")
print(f"   PnL:      ${pnl:+.2f} ({pnl_pct:+.1%})")
print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
print(f"   Trades:   {trades}")
print(f"   Max DD:   {dd:.1%}")
print(f"\n📊 Random baseline: ~-$983, 15.8% WR")

if pnl > 0:
    print(f"\n✅ Agent profitable — learned to trade BTC spot!")
elif pnl > -200:
    print(f"\n⚠️  Agent losing less than random — learning")
else:
    print(f"\n❌ Agent struggling — needs more steps")

print(f"\n💾 Saved: crypto_agent_best.zip")
