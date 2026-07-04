"""
PPO Training Script — Technical Indicator Observations (44 features)
====================================================================
Trains a new PPO model on CryptoTechRREnv with technical indicator
features aligned with the reasoning engine.

Usage:
    cd /Users/moura/trading-indicator/ai-model/crypto
    python3 crypto_train_15m.py
"""
# MTF_15M_RETRAIN

import sys, os  # MTF_15M_RETRAIN
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # MTF_15M_RETRAIN

import pandas as pd  # MTF_15M_RETRAIN
import numpy as np  # MTF_15M_RETRAIN
from binance.client import Client  # MTF_15M_RETRAIN
from stable_baselines3 import PPO  # MTF_15M_RETRAIN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback  # MTF_15M_RETRAIN
from crypto_env_v2 import CryptoTechRREnv  # MTF_15M_RETRAIN

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"  # MTF_15M_RETRAIN
MODEL_SAVE = "crypto_mtf_tech_best.zip"  # MTF_15M_RETRAIN
DATASET_PKL = "btc_tech_mtf.pkl"  # MTF_15M_RETRAIN
TOTAL_TIMESTEPS = 2_000_000  # MTF_15M_RETRAIN
N_STEPS = 2048  # MTF_15M_RETRAIN
BATCH_SIZE = 256  # MTF_15M_RETRAIN
LEARNING_RATE = 0.0002  # MTF_15M_RETRAIN
N_EPOCHS = 10  # MTF_15M_RETRAIN
ENT_COEF = 0.01  # MTF_15M_RETRAIN
NET_ARCH = [256, 256, 128]  # MTF_15M_RETRAIN
EVAL_FREQ = 10_000  # MTF_15M_RETRAIN


# ── Progress callback ────────────────────────────────────────────────────────
class ProgressCallback(BaseCallback):  # MTF_15M_RETRAIN
    def __init__(self, print_freq=50_000):  # MTF_15M_RETRAIN
        super().__init__()  # MTF_15M_RETRAIN
        self.print_freq = print_freq  # MTF_15M_RETRAIN

    def _on_step(self):  # MTF_15M_RETRAIN
        if self.num_timesteps % self.print_freq < self.locals.get('n_steps', N_STEPS):  # MTF_15M_RETRAIN
            print(f"  Step {self.num_timesteps:,} / {TOTAL_TIMESTEPS:,} "  # MTF_15M_RETRAIN
                  f"({self.num_timesteps / TOTAL_TIMESTEPS:.0%})")  # MTF_15M_RETRAIN
        return True  # MTF_15M_RETRAIN


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_training_data():  # MTF_15M_RETRAIN
    """Fetch 1H + 4H + 1D + 1W BTC data for training."""  # MTF_15M_RETRAIN

    # Check for cached dataset  # MTF_15M_RETRAIN
    if os.path.exists(DATASET_PKL):  # MTF_15M_RETRAIN
        print(f"📦 Loading cached dataset from {DATASET_PKL}")  # MTF_15M_RETRAIN
        return pd.read_pickle(DATASET_PKL)  # MTF_15M_RETRAIN

    print(f"📥 Fetching historical {SYMBOL} data...")  # MTF_15M_RETRAIN
    client = Client("", "")  # MTF_15M_RETRAIN
    client.ping = lambda: None  # MTF_15M_RETRAIN

    def fetch_tf(interval, start, limit=1000):  # MTF_15M_RETRAIN
        all_klines = []  # MTF_15M_RETRAIN
        while True:  # MTF_15M_RETRAIN
            klines = client.get_historical_klines(  # MTF_15M_RETRAIN
                SYMBOL, interval, start, limit=limit)  # MTF_15M_RETRAIN
            if not klines:  # MTF_15M_RETRAIN
                break  # MTF_15M_RETRAIN
            all_klines.extend(klines)  # MTF_15M_RETRAIN
            start = klines[-1][0] + 1  # MTF_15M_RETRAIN
            if len(klines) < limit:  # MTF_15M_RETRAIN
                break  # MTF_15M_RETRAIN
            if len(all_klines) % 5000 == 0:  # MTF_15M_RETRAIN
                print(f"  {len(all_klines)} bars...", end="\r")  # MTF_15M_RETRAIN
        return all_klines  # MTF_15M_RETRAIN

    def to_df(klines):  # MTF_15M_RETRAIN
        df = pd.DataFrame(klines, columns=[  # MTF_15M_RETRAIN
            'ts', 'open', 'high', 'low', 'close', 'volume',  # MTF_15M_RETRAIN
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'])  # MTF_15M_RETRAIN
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)  # MTF_15M_RETRAIN
        for col in ['open', 'high', 'low', 'close', 'volume']:  # MTF_15M_RETRAIN
            df[col] = df[col].astype(float)  # MTF_15M_RETRAIN
        return df[['ts', 'open', 'high', 'low', 'close', 'volume']].set_index('ts')  # MTF_15M_RETRAIN

    start_date = "1 Jan, 2020"  # MTF_15M_RETRAIN

    print("  Fetching 1H bars...")  # MTF_15M_RETRAIN
    df_1h = to_df(fetch_tf(Client.KLINE_INTERVAL_1HOUR, start_date))  # MTF_15M_RETRAIN
    print(f"  1H: {len(df_1h)} bars")  # MTF_15M_RETRAIN

    print("  Fetching 4H bars...")  # MTF_15M_RETRAIN
    df_4h = to_df(fetch_tf(Client.KLINE_INTERVAL_4HOUR, start_date))  # MTF_15M_RETRAIN
    print(f"  4H: {len(df_4h)} bars")  # MTF_15M_RETRAIN

    print("  Fetching 1D bars...")  # MTF_15M_RETRAIN
    df_1d = to_df(fetch_tf(Client.KLINE_INTERVAL_1DAY, start_date))  # MTF_15M_RETRAIN
    print(f"  1D: {len(df_1d)} bars")  # MTF_15M_RETRAIN

    print("  Fetching 1W bars...")  # MTF_15M_RETRAIN
    df_1w = to_df(fetch_tf(Client.KLINE_INTERVAL_1WEEK, start_date))  # MTF_15M_RETRAIN
    print(f"  1W: {len(df_1w)} bars")  # MTF_15M_RETRAIN

    # Build MTF dataset with 1H as base  # MTF_15M_RETRAIN
    df_4h_ff = df_4h.reindex(df_1h.index, method='ffill').add_prefix('h4_')  # MTF_15M_RETRAIN
    df_1d_ff = df_1d.reindex(df_1h.index, method='ffill').add_prefix('d1_')  # MTF_15M_RETRAIN
    df_1w_ff = df_1w.reindex(df_1h.index, method='ffill').add_prefix('w1_')  # MTF_15M_RETRAIN

    df = pd.concat([df_1h, df_4h_ff, df_1d_ff, df_1w_ff], axis=1).dropna()  # MTF_15M_RETRAIN
    df = df.reset_index().rename(columns={'ts': 'Datetime'})  # MTF_15M_RETRAIN

    print(f"\n✅ MTF dataset: {len(df)} bars "  # MTF_15M_RETRAIN
          f"({df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]})")  # MTF_15M_RETRAIN

    df.to_pickle(DATASET_PKL)  # MTF_15M_RETRAIN
    print(f"💾 Saved to {DATASET_PKL}")  # MTF_15M_RETRAIN
    return df  # MTF_15M_RETRAIN


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # MTF_15M_RETRAIN
    print("🚀 PPO Training — CryptoTechRREnv (44 features)")  # MTF_15M_RETRAIN
    print(f"   Model:      {MODEL_SAVE}")  # MTF_15M_RETRAIN
    print(f"   Timesteps:  {TOTAL_TIMESTEPS:,}")  # MTF_15M_RETRAIN
    print(f"   Net arch:   {NET_ARCH}")  # MTF_15M_RETRAIN
    print(f"   LR:         {LEARNING_RATE}")  # MTF_15M_RETRAIN
    print(f"   Ent coef:   {ENT_COEF}")  # MTF_15M_RETRAIN
    print()  # MTF_15M_RETRAIN

    # Fetch data  # MTF_15M_RETRAIN
    df = fetch_training_data()  # MTF_15M_RETRAIN

    # Split: 80% train, 20% eval  # MTF_15M_RETRAIN
    split = int(len(df) * 0.8)  # MTF_15M_RETRAIN
    df_train = df.iloc[:split].reset_index(drop=True)  # MTF_15M_RETRAIN
    df_eval = df.iloc[split:].reset_index(drop=True)  # MTF_15M_RETRAIN
    print(f"📊 Train: {len(df_train)} bars | Eval: {len(df_eval)} bars")  # MTF_15M_RETRAIN

    # Create environments  # MTF_15M_RETRAIN
    train_env = CryptoTechRREnv(df_train, sl_pct=0.025)  # MTF_15M_RETRAIN: match live SL
    eval_env = CryptoTechRREnv(df_eval, sl_pct=0.025)  # MTF_15M_RETRAIN

    # Verify observation space  # MTF_15M_RETRAIN
    obs, _ = train_env.reset()  # MTF_15M_RETRAIN
    print(f"✅ Observation shape: {obs.shape} (expected 44)")  # MTF_15M_RETRAIN
    assert obs.shape[0] == 44, f"Expected 44 features, got {obs.shape[0]}"  # MTF_15M_RETRAIN

    # Create PPO model  # MTF_15M_RETRAIN
    model = PPO(  # MTF_15M_RETRAIN
        "MlpPolicy",  # MTF_15M_RETRAIN
        train_env,  # MTF_15M_RETRAIN
        n_steps=N_STEPS,  # MTF_15M_RETRAIN
        batch_size=BATCH_SIZE,  # MTF_15M_RETRAIN
        learning_rate=LEARNING_RATE,  # MTF_15M_RETRAIN
        n_epochs=N_EPOCHS,  # MTF_15M_RETRAIN
        ent_coef=ENT_COEF,  # MTF_15M_RETRAIN
        policy_kwargs={"net_arch": NET_ARCH},  # MTF_15M_RETRAIN
        verbose=0,  # MTF_15M_RETRAIN
    )  # MTF_15M_RETRAIN
    print(f"✅ PPO model created: {NET_ARCH}")  # MTF_15M_RETRAIN

    # Callbacks  # MTF_15M_RETRAIN
    eval_callback = EvalCallback(  # MTF_15M_RETRAIN
        eval_env,  # MTF_15M_RETRAIN
        best_model_save_path="./",  # MTF_15M_RETRAIN
        log_path="./logs_tech/",  # MTF_15M_RETRAIN
        eval_freq=EVAL_FREQ,  # MTF_15M_RETRAIN
        n_eval_episodes=1,  # MTF_15M_RETRAIN
        deterministic=True,  # MTF_15M_RETRAIN
        render=False,  # MTF_15M_RETRAIN
    )  # MTF_15M_RETRAIN

    progress_cb = ProgressCallback(print_freq=50_000)  # MTF_15M_RETRAIN

    # Train  # MTF_15M_RETRAIN
    print(f"\n🧠 Training for {TOTAL_TIMESTEPS:,} timesteps...\n")  # MTF_15M_RETRAIN
    model.learn(  # MTF_15M_RETRAIN
        total_timesteps=TOTAL_TIMESTEPS,  # MTF_15M_RETRAIN
        callback=[eval_callback, progress_cb],  # MTF_15M_RETRAIN
    )  # MTF_15M_RETRAIN

    # Save final model  # MTF_15M_RETRAIN
    model.save(MODEL_SAVE)  # MTF_15M_RETRAIN
    print(f"\n✅ Model saved: {MODEL_SAVE}")  # MTF_15M_RETRAIN

    # Quick eval  # MTF_15M_RETRAIN
    print("\n📊 Final evaluation on eval set...")  # MTF_15M_RETRAIN
    obs, _ = eval_env.reset()  # MTF_15M_RETRAIN
    done = False  # MTF_15M_RETRAIN
    while not done:  # MTF_15M_RETRAIN
        action, _ = model.predict(obs, deterministic=True)  # MTF_15M_RETRAIN
        obs, _, done, _, info = eval_env.step(action)  # MTF_15M_RETRAIN

    print(f"   PnL:    ${info.get('total_pnl', 0):+.2f} "  # MTF_15M_RETRAIN
          f"({info.get('total_pnl_pct', 0):+.1%})")  # MTF_15M_RETRAIN
    print(f"   WR:     {info.get('win_rate', 0):.1%}")  # MTF_15M_RETRAIN
    print(f"   Trades: {info.get('total_trades', 0)}")  # MTF_15M_RETRAIN
    print(f"   DD:     {info.get('max_drawdown', 0):.1%}")  # MTF_15M_RETRAIN
    print(f"\n🚀 Done. Model ready at {MODEL_SAVE}")  # MTF_15M_RETRAIN
