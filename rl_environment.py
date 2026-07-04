"""
NQ Futures RL Trading Environment
==================================
Wraps the ICT backtest logic into a Gymnasium environment
so a reinforcement learning agent can learn to trade.

State:  15 market features per bar
Action: 0=HOLD, 1=BUY, 2=SELL
Reward: Based on realized PnL + drawdown penalty
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class NQTradingEnv(gym.Env):
    """
    NQ Futures Trading Environment for Reinforcement Learning.

    The agent observes market features and decides:
    - 0: HOLD (do nothing)
    - 1: BUY (go long)
    - 2: SELL (go short)

    It learns by trial and error what combinations of
    market conditions lead to profitable trades.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 sl_pts: float = 30.0, tp_pts: float = 30.0,
                 cooldown_bars: int = 6, max_daily_trades: int = 5):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.sl_pts = sl_pts
        self.tp_pts = tp_pts
        self.cooldown_bars = cooldown_bars
        self.max_daily_trades = max_daily_trades

        # ── Action space: HOLD / BUY / SELL ──────────────────────────────
        self.action_space = spaces.Discrete(3)

        # ── Observation space: 15 normalized features ─────────────────────
        # All values normalized to [-1, 1] or [0, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(15,), dtype=np.float32
        )

        # Start index — need at least 50 bars of history
        self.start_idx = 50
        self.current_idx = self.start_idx
        self._precompute_features()

    def _precompute_features(self):
        """Precompute all technical features for every bar."""
        df = self.df
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # EMAs for trend detection
        ema9   = close.ewm(span=9).mean()
        ema21  = close.ewm(span=21).mean()
        ema50  = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, 0.001))

        # ATR (volatility)
        tr  = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Volume ratio
        vol_avg   = vol.rolling(20).mean()
        vol_ratio = (vol / vol_avg.replace(0, 1)).clip(0, 5)

        # Price position in recent range
        hi20 = high.rolling(20).max()
        lo20 = low.rolling(20).min()
        price_pos = (close - lo20) / (hi20 - lo20 + 1e-8)

        # Candle body direction
        body = (close - df['open']) / (atr + 1e-8)

        # Store as numpy array for fast access
        self.features = np.column_stack([
            # Trend features (normalized)
            ((ema9 - ema21) / (atr + 1e-8)).clip(-3, 3) / 3,        # 0: short-term momentum
            ((ema21 - ema50) / (atr + 1e-8)).clip(-3, 3) / 3,       # 1: medium-term momentum
            ((ema50 - ema200) / (atr + 1e-8)).clip(-3, 3) / 3,      # 2: long-term trend
            # RSI normalized to [-1, 1]
            (rsi / 50 - 1).clip(-1, 1),                              # 3: RSI
            # Volume
            (vol_ratio / 2.5 - 1).clip(-1, 1),                      # 4: volume spike
            # Price position
            price_pos * 2 - 1,                                       # 5: position in range
            # Candle body
            body.clip(-2, 2) / 2,                                    # 6: candle direction
            # ATR normalized (volatility)
            (atr / close * 100).clip(0, 2) - 1,                     # 7: volatility
            # Session encoding (one-hot-ish)
            np.zeros(len(df)),                                        # 8: is_london
            np.zeros(len(df)),                                        # 9: is_ny_open
            np.zeros(len(df)),                                        # 10: is_ny_pm
            # Time of day
            np.zeros(len(df)),                                        # 11: hour sin
            np.zeros(len(df)),                                        # 12: hour cos
            # Recent returns
            close.pct_change(3).clip(-0.02, 0.02) / 0.02,           # 13: 15min return
            close.pct_change(12).clip(-0.05, 0.05) / 0.05,          # 14: 1hr return
        ]).astype(np.float32)

        # Fill session and time features
        if 'datetime' in df.columns or hasattr(df.index, 'hour'):
            try:
                idx = df.index if hasattr(df.index, 'hour') else pd.to_datetime(df['datetime'])
                hours = idx.hour + idx.minute / 60
                self.features[:, 8] = ((hours >= 9) & (hours < 13.5)).astype(float)   # london
                self.features[:, 9] = ((hours >= 13.5) & (hours < 15)).astype(float)  # ny open
                self.features[:, 10] = ((hours >= 17.5) & (hours < 19)).astype(float) # ny pm
                self.features[:, 11] = np.sin(2 * np.pi * hours / 24)
                self.features[:, 12] = np.cos(2 * np.pi * hours / 24)
            except:
                pass

        # Replace NaN with 0
        self.features = np.nan_to_num(self.features, 0.0)

    def _get_obs(self):
        return self.features[self.current_idx].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx  = self.start_idx
        self.capital      = self.initial_capital
        self.position     = 0        # 0=flat, 1=long, -1=short
        self.entry_price  = 0.0
        self.last_trade_bar = -999
        self.daily_trades = {}
        self.total_pnl    = 0.0
        self.wins         = 0
        self.losses       = 0
        self.max_capital  = self.initial_capital
        self.max_drawdown = 0.0
        return self._get_obs(), {}

    def step(self, action):
        bar = self.df.iloc[self.current_idx]
        close = float(bar['close'])
        reward = 0.0
        info   = {}

        # ── Check if already in trade ─────────────────────────────────────
        if self.position != 0:
            # Check SL/TP hit
            if self.position == 1:   # Long
                if float(bar['low']) <= self.entry_price - self.sl_pts:
                    reward = -1.0
                    self.losses += 1
                    self.capital -= self.sl_pts * 2  # MNQ = $2/pt
                    self.position = 0
                elif float(bar['high']) >= self.entry_price + self.tp_pts:
                    reward = +1.0
                    self.wins += 1
                    self.capital += self.tp_pts * 2
                    self.position = 0
            elif self.position == -1:  # Short
                if float(bar['high']) >= self.entry_price + self.sl_pts:
                    reward = -1.0
                    self.losses += 1
                    self.capital -= self.sl_pts * 2
                    self.position = 0
                elif float(bar['low']) <= self.entry_price - self.tp_pts:
                    reward = +1.0
                    self.wins += 1
                    self.capital += self.tp_pts * 2
                    self.position = 0

        # ── Take new action if flat ────────────────────────────────────────
        elif action in (1, 2):
            cooldown_ok = (self.current_idx - self.last_trade_bar) >= self.cooldown_bars
            day = str(self.current_idx // 78)  # ~78 bars per trading day at 5m
            daily_ok = self.daily_trades.get(day, 0) < self.max_daily_trades

            if cooldown_ok and daily_ok:
                self.position    = 1 if action == 1 else -1
                self.entry_price = close
                self.last_trade_bar = self.current_idx
                self.daily_trades[day] = self.daily_trades.get(day, 0) + 1
                reward = -0.01  # small cost for entering (spread/commission)

        # ── Drawdown tracking ─────────────────────────────────────────────
        self.max_capital  = max(self.max_capital, self.capital)
        drawdown = (self.max_capital - self.capital) / self.max_capital
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Penalize large drawdowns
        if drawdown > 0.1:
            reward -= 0.5

        # ── Advance ───────────────────────────────────────────────────────
        self.current_idx += 1
        done = self.current_idx >= len(self.df) - 1

        if done:
            # Final reward: Sharpe-like score
            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
            reward += pnl_pct * 10 + wr * 2

            info = {
                "total_pnl": self.capital - self.initial_capital,
                "win_rate": wr,
                "wins": self.wins,
                "losses": self.losses,
                "max_drawdown": self.max_drawdown,
            }

        obs = self._get_obs() if not done else self.features[self.current_idx - 1].copy()
        return obs, reward, done, False, info

    def render(self):
        bar = self.df.iloc[self.current_idx]
        wr  = self.wins / max(1, self.wins + self.losses)
        print(f"Bar {self.current_idx} | Price: {bar['close']:.2f} | "
              f"Capital: ${self.capital:.0f} | "
              f"Position: {'LONG' if self.position==1 else 'SHORT' if self.position==-1 else 'FLAT'} | "
              f"WR: {wr:.1%} ({self.wins}W/{self.losses}L)")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    print("📥 Downloading NQ data for environment test...")
    df = yf.download("NQ=F", period="60d", interval="5m", progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.dropna().reset_index()

    print(f"✅ {len(df)} bars loaded")

    # Create environment
    env = NQTradingEnv(df)
    obs, _ = env.reset()

    print(f"\n✅ Environment created")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Sample observation: {obs[:5].round(3)}")

    # Random agent test — 1 full episode
    print("\n🎲 Running random agent test (1 episode)...")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()  # random action
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"\n📊 Random Agent Results:")
    print(f"   Steps: {steps}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final PnL: ${info.get('total_pnl', 0):.0f}")
    print(f"   Win rate: {info.get('win_rate', 0):.1%}")
    print(f"   Wins: {info.get('wins', 0)} | Losses: {info.get('losses', 0)}")
    print(f"   Max drawdown: {info.get('max_drawdown', 0):.1%}")
    print("\n✅ Environment working correctly")
