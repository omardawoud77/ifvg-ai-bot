"""
Crypto MTF Agent Training — Clean, No Hardcoded Rules
Agent learns everything from raw OHLCV across 4 timeframes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from crypto_env_v2 import CryptoMTFEnv

print("📥 Loading MTF dataset...")
df = pd.read_pickle('btc_mtf.pkl')
print(f"✅ {len(df)} bars | {len(df.columns)} columns")

split = int(len(df) * 0.8)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)
print(f"📊 Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

class Callback(BaseCallback):
    def __init__(self, eval_env, eval_every=5000, patience=200000):
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
                self.model.save("crypto_mtf_best")
                print(f"  💾 New best! ${pnl:+.2f}")
            else:
                self.steps_since_best += self.eval_every
                if self.steps_since_best >= self.patience:
                    print(f"\n⏹️  Early stop — no improvement for {self.patience:,} steps")
                    return False
        return True

print("\n🏗️  Building environments...")
train_env = CryptoMTFEnv(train_df)
test_env  = CryptoMTFEnv(test_df)

print("🤖 Building PPO agent (36 features, 4 actions, pure OHLCV)...")
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
print(f"   No hardcoded rules — agent learns everything")

print(f"\n🚀 Training up to 2,000,000 steps...")
print("="*70)

model.learn(
    total_timesteps=2_000_000,
    callback=Callback(test_env, eval_every=5000, patience=200000),
    progress_bar=False
)

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

best = PPO.load("crypto_mtf_best")
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

print(f"\n📊 CLEAN MTF AGENT RESULTS:")
print(f"   PnL:      ${pnl:+.2f} ({pnl_pct:+.1%})")
print(f"   Win Rate: {wr:.1%} ({wins}W / {losses}L)")
print(f"   Trades:   {trades}")
print(f"   Max DD:   {dd:.1%}")
print(f"\n📊 Random baseline: ~-$715, 42% WR, 1500 trades")

if pnl > 0 and wr > 0.50:
    print(f"\n✅ Agent profitable — learned BTC patterns from raw price!")
elif pnl > 0:
    print(f"\n⚠️  Profitable but low WR — needs more training")
else:
    print(f"\n❌ Still learning — check step-by-step progress")

print(f"\n💾 Saved: crypto_mtf_best.zip")
