"""
RL Live Agent — Online Learning from Live Market
=================================================
Runs alongside the ICT live bot on Railway.
Observes live NQ 5m bars, paper trades, and learns continuously.

Architecture:
- Pulls fresh 5m bar every 5 minutes from Yahoo
- Maintains rolling buffer of last 2000 bars (replay memory)
- Agent acts on each new bar (paper trade)
- Every 50 bars: trains PPO on random sample from replay buffer
- Saves model checkpoint every 100 bars
- Logs all paper trades + learning progress

Modes:
- SHADOW: observes and learns, no output to live bot
- This is Mode 1 — safe, pure learning
"""

import os
import time
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_environment_v3 import NQTradingEnvV3

# ── Config ────────────────────────────────────────────────────────────────────
BUFFER_SIZE      = 2000    # bars kept in replay memory
MIN_BARS_TRAIN   = 400     # minimum bars before first training
RETRAIN_EVERY    = 50      # retrain every N new bars
CHECKPOINT_EVERY = 100     # save model every N bars
FETCH_INTERVAL   = 15      # seconds between bar fetches (15s)
MODEL_PATH       = "rl_live_agent.zip"
BUFFER_PATH      = "rl_live_buffer.pkl"
LOG_PATH         = "rl_live_agent.log"
TRADES_PATH      = "rl_live_trades.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
class LiveAgentState:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)  # rolling bar buffer
        self.bars_seen = 0
        self.bars_since_retrain = 0
        self.paper_trades = []
        self.capital = 10000.0
        self.position = 0       # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.entry_bar = 0
        self.wins = 0
        self.losses = 0
        self.model = None
        self.last_bar_time = None

    @property
    def wr(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def save(self):
        with open(BUFFER_PATH, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'bars_seen': self.bars_seen,
                'capital': self.capital,
                'wins': self.wins,
                'losses': self.losses,
                'paper_trades': self.paper_trades,
            }, f)

    def load(self):
        if os.path.exists(BUFFER_PATH):
            with open(BUFFER_PATH, 'rb') as f:
                data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=BUFFER_SIZE)
            self.bars_seen = data['bars_seen']
            self.capital = data['capital']
            self.wins = data['wins']
            self.losses = data['losses']
            self.paper_trades = data['paper_trades']
            log.info(f"📂 Loaded state: {len(self.buffer)} bars, "
                     f"{self.bars_seen} total seen, "
                     f"capital ${self.capital:.0f}")
            return True
        return False


# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_latest_bars(n=300):
    """Fetch latest N 5m bars from Yahoo Finance."""
    try:
        df = yf.download("NQ=F", period="7d", interval="1m",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df.dropna()
        # Resample 1m → 5m so features stay compatible with v3 training
        df = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        df = df.tail(n)
        df = df.reset_index()
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'Datetime'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'Datetime'})
        return df
    except Exception as e:
        log.error(f"❌ Fetch error: {e}")
        return None


# ── Feature extraction (simplified for online use) ────────────────────────────
def get_obs_from_buffer(state):
    """Build observation from current buffer state."""
    if len(state.buffer) < 50:
        return None

    df = pd.DataFrame(list(state.buffer))

    # Ensure correct column names
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            return None

    # Build minimal env for feature extraction
    try:
        env = NQTradingEnvV3(df, point_value=2.0)
        env.reset()
        # Fast-forward to last bar
        env.current_idx = len(df) - 1
        # Manually set position state
        env.position = state.position
        env.entry_price = state.entry_price
        env.bars_held = state.bars_seen - state.entry_bar if state.position != 0 else 0
        env.unrealized_pnl = 0.0
        if state.position != 0:
            close = float(df['close'].iloc[-1])
            if state.position == 1:
                env.unrealized_pnl = (close - state.entry_price) * 2.0
            else:
                env.unrealized_pnl = (state.entry_price - close) * 2.0
        env.capital = state.capital
        env.wins = state.wins
        env.losses = state.losses

        obs = env._get_obs(len(df) - 1)
        return obs
    except Exception as e:
        log.warning(f"⚠️  Obs extraction error: {e}")
        return None


# ── Paper trading execution ───────────────────────────────────────────────────
def execute_paper_action(state, action, current_bar):
    """Execute paper trade action and return reward."""
    close = float(current_bar['close'])
    reward = 0.0
    trade_log = None

    if state.position == 0:  # FLAT
        if action == 1:  # BUY
            state.position = 1
            state.entry_price = close + 0.25  # commission
            state.entry_bar = state.bars_seen
            log.info(f"📈 PAPER LONG @ {close:.2f} | Capital: ${state.capital:.0f}")

        elif action == 2:  # SELL
            state.position = -1
            state.entry_price = close - 0.25
            state.entry_bar = state.bars_seen
            log.info(f"📉 PAPER SHORT @ {close:.2f} | Capital: ${state.capital:.0f}")

    elif state.position != 0:  # IN TRADE
        bars_held = state.bars_seen - state.entry_bar

        # Unrealized PnL
        if state.position == 1:
            upnl = (close - state.entry_price) * 2.0
        else:
            upnl = (state.entry_price - close) * 2.0

        # Per-bar reward
        reward = upnl * 0.001

        # Force close after 48 bars
        should_close = (action == 3) or (bars_held >= 48)

        if should_close:
            pnl = upnl
            state.capital += pnl

            if pnl > 0:
                state.wins += 1
                result = "WIN"
            else:
                state.losses += 1
                result = "LOSS"

            reward += pnl * 0.1
            if bars_held >= 48:
                reward -= 20  # timeout penalty

            trade_log = {
                "time": str(current_bar.get('Datetime', datetime.now())),
                "direction": "long" if state.position == 1 else "short",
                "entry": state.entry_price,
                "exit": close,
                "pnl": round(pnl, 2),
                "bars_held": bars_held,
                "result": result,
                "capital": round(state.capital, 2),
                "wr": round(state.wr, 3),
            }

            log.info(f"{'✅' if pnl > 0 else '❌'} CLOSED {result} | "
                     f"PnL: ${pnl:+.0f} | "
                     f"Bars: {bars_held} | "
                     f"Capital: ${state.capital:.0f} | "
                     f"WR: {state.wr:.1%} ({state.wins}W/{state.losses}L)")

            state.paper_trades.append(trade_log)
            state.position = 0
            state.entry_price = 0.0

            # Save trades log
            with open(TRADES_PATH, 'w') as f:
                json.dump(state.paper_trades, f, indent=2)

    return reward, trade_log


# ── Online training ───────────────────────────────────────────────────────────
def retrain_on_buffer(state):
    """Retrain model on current buffer contents."""
    if len(state.buffer) < MIN_BARS_TRAIN:
        log.info(f"⏳ Buffer too small ({len(state.buffer)}/{MIN_BARS_TRAIN}) — skipping retrain")
        return

    log.info(f"🔄 Retraining on {len(state.buffer)} bars...")

    df = pd.DataFrame(list(state.buffer))

    # Need at least 400 bars for a meaningful env
    if len(df) < 400:
        return

    try:
        split = int(len(df) * 0.85)
        train_df = df.iloc[:split].reset_index(drop=True)

        env = NQTradingEnvV3(train_df, point_value=2.0)

        if state.model is None:
            bootstrap = getattr(state, '_bootstrap_model', None)
            if bootstrap and os.path.exists(bootstrap):
                log.info(f"🤖 Loading bootstrap model: {bootstrap}")
                state.model = PPO.load(bootstrap, env=env)
                log.info("✅ Bootstrap model loaded — agent starts with prior knowledge!")
            else:
                log.info("🤖 Building new PPO model from scratch...")
                state.model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=1e-4,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.1,
                    ent_coef=0.02,
                    verbose=0,
                    policy_kwargs=dict(net_arch=[256, 256, 128])
                )
        else:
            # Update env for continued training
            state.model.set_env(env)

        # Short training burst — 2048 steps
        state.model.learn(
            total_timesteps=2048,
            reset_num_timesteps=False,  # continue from where we left off
            progress_bar=False
        )

        log.info(f"✅ Retrain complete | "
                 f"Bars seen: {state.bars_seen} | "
                 f"WR: {state.wr:.1%} | "
                 f"Capital: ${state.capital:.0f}")

        state.bars_since_retrain = 0

    except Exception as e:
        log.error(f"❌ Retrain error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    log.info("="*60)
    log.info("🚀 RL Live Agent Starting")
    log.info("   Mode: SHADOW (paper trading + online learning)")
    log.info("   Instrument: NQ Futures (MNQ equivalent)")
    log.info("="*60)

    state = LiveAgentState()

    # Load existing state if available
    resumed = state.load()
    if resumed:
        log.info(f"▶️  Resuming from checkpoint")
    else:
        log.info("🆕 Starting fresh")

    # Bootstrap from best trained model if available
    bootstrap_model = None
    if os.path.exists(MODEL_PATH):
        bootstrap_model = MODEL_PATH
        log.info(f"📦 Will load checkpoint: {MODEL_PATH}")
    elif os.path.exists("rl_agent_v3_best.zip"):
        bootstrap_model = "rl_agent_v3_best.zip"
        log.info(f"📦 Will bootstrap from v3 best model")
    state._bootstrap_model = bootstrap_model

    last_processed_time = None
    bars_since_checkpoint = 0

    while True:
        try:
            # ── Fetch latest data ──────────────────────────────────────
            log.info("📥 Fetching latest bars...")
            df = fetch_latest_bars(300)

            if df is None or len(df) < 50:
                log.warning("⚠️  No data — retrying in 60s")
                time.sleep(60)
                continue

            # ── Find new bars ──────────────────────────────────────────
            latest_time = df['Datetime'].iloc[-1] if 'Datetime' in df.columns else df.index[-1]

            if last_processed_time is None:
                # First run — seed buffer with historical bars
                log.info(f"🌱 Seeding buffer with {len(df)} bars...")
                for _, row in df.iterrows():
                    state.buffer.append(row.to_dict())
                last_processed_time = latest_time
                state.bars_seen = len(df)
                log.info(f"✅ Buffer seeded: {len(state.buffer)} bars")

            else:
                # Find bars newer than last processed
                if 'Datetime' in df.columns:
                    new_bars = df[df['Datetime'] > last_processed_time]
                else:
                    new_bars = df[df.index > last_processed_time]

                if len(new_bars) == 0:
                    log.info(f"⏸️  No new bars — waiting {FETCH_INTERVAL}s")
                    time.sleep(FETCH_INTERVAL)
                    continue

                log.info(f"📊 {len(new_bars)} new bar(s) received")

                for _, bar in new_bars.iterrows():
                    bar_dict = bar.to_dict()
                    state.buffer.append(bar_dict)
                    state.bars_seen += 1
                    state.bars_since_retrain += 1
                    bars_since_checkpoint += 1

                    # ── Get agent action ───────────────────────────────
                    if state.model is not None and len(state.buffer) >= 50:
                        obs = get_obs_from_buffer(state)
                        if obs is not None:
                            action, _ = state.model.predict(obs, deterministic=False)
                            action = int(action)
                        else:
                            action = 0  # HOLD if obs fails
                    else:
                        action = 0  # HOLD until model is trained

                    # ── Execute paper trade ────────────────────────────
                    reward, trade = execute_paper_action(state, action, bar_dict)

                    # ── Log status every bar ───────────────────────────
                    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
                    pos_names = {0: "FLAT", 1: "LONG", -1: "SHORT"}
                    log.info(f"Bar {state.bars_seen:5d} | "
                             f"Action: {action_names[action]:5s} | "
                             f"Pos: {pos_names[state.position]:5s} | "
                             f"Capital: ${state.capital:.0f} | "
                             f"Buffer: {len(state.buffer)}")

                last_processed_time = latest_time

            # ── Retrain if due ─────────────────────────────────────────
            if state.bars_since_retrain >= RETRAIN_EVERY:
                retrain_on_buffer(state)

            # ── Save checkpoint if due ─────────────────────────────────
            if bars_since_checkpoint >= CHECKPOINT_EVERY:
                state.save()
                if state.model is not None:
                    state.model.save(MODEL_PATH)
                    log.info(f"💾 Checkpoint saved | "
                             f"Bars: {state.bars_seen} | "
                             f"WR: {state.wr:.1%} | "
                             f"Capital: ${state.capital:.0f}")
                bars_since_checkpoint = 0

            # ── Status summary ─────────────────────────────────────────
            total_trades = state.wins + state.losses
            if total_trades > 0:
                log.info(f"\n{'='*50}")
                log.info(f"📊 LIVE AGENT STATUS")
                log.info(f"   Bars seen:    {state.bars_seen:,}")
                log.info(f"   Buffer size:  {len(state.buffer):,}")
                log.info(f"   Paper trades: {total_trades}")
                log.info(f"   Win rate:     {state.wr:.1%}")
                log.info(f"   Capital:      ${state.capital:.0f} "
                         f"(P&L: ${state.capital - 10000:+.0f})")
                log.info(f"{'='*50}\n")

            # ── Wait for next bar ──────────────────────────────────────
            log.info(f"⏳ Waiting {FETCH_INTERVAL}s for next bar...")
            time.sleep(FETCH_INTERVAL)

        except KeyboardInterrupt:
            log.info("🛑 Shutting down — saving state...")
            state.save()
            if state.model is not None:
                state.model.save(MODEL_PATH)
            log.info("✅ State saved. Goodbye.")
            break

        except Exception as e:
            log.error(f"❌ Main loop error: {e}")
            log.info("🔄 Recovering in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    run()
