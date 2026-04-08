"""
Crypto RL Trading Environment
==============================
Adapted from NQTradingEnvV3 for crypto markets.

Key differences from NQ:
- 24/7 trading — no session gates
- Price-relative features (% returns, not points)
- Trade size in USDT, not contracts
- Binance testnet compatible
- Supports BTC, ETH, SOL (same env, different symbol)

Observation (28 features):
  [0-5]   Price structure: returns (1,3,6,12,24,48 bars)
  [6-8]   Candle: body ratio, upper wick, lower wick
  [9-12]  HTF bias: 4H, daily, weekly (EMA-based)
  [13-15] ICT: FVG present+dist, OB proximity
  [16-18] Momentum: RSI, momentum, volatility
  [19-21] Volume: vol ratio, vol trend, buy pressure
  [22-24] Position state: in_trade, direction, unrealized_pnl_pct
  [25-26] Bars held, max_drawdown_pct
  [27]    Time of day (hour_sin)

Actions:
  0 = HOLD
  1 = BUY (long)
  2 = SELL (short)
  3 = CLOSE

Reward:
  - Per bar in trade: unrealized PnL % change
  - On close: realized PnL %
  - Drawdown penalty: -1.0 per bar if DD > 3%
  - Forced close penalty: -5 if held > 96 bars (8 hours)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _calc_htf_bias(close, i):
    """HTF bias for crypto 5m bars. 4H=48, Daily=288, Weekly=2016."""
    s = close.iloc[max(0, i-2500):i+1]
    if len(s) < 100:
        return 0.0, 0.0, 0.0

    def bias(fast, slow):
        f = _ema(s, fast).iloc[-1]
        sl = _ema(s, slow).iloc[-1]
        diff = (f - sl) / (abs(sl) + 1e-8)
        return float(np.clip(diff * 100, -1, 1))

    h4     = bias(48, 96)
    daily  = bias(288, 576)
    weekly = bias(2016, 4032)
    return h4, daily, weekly


def _find_fvg(high, low, close_val, i, df_high, df_low):
    """Find nearest FVG. Returns (present, distance_pct)."""
    lookback = min(50, i)
    h = df_high.iloc[i-lookback:i].values
    l = df_low.iloc[i-lookback:i].values

    best_dist = 999.0
    found = 0.0

    for j in range(2, len(h)-1):
        # Bullish FVG
        gap_lo = h[-j-1]
        gap_hi = l[-j+1] if -j+1 < 0 else l[len(l)-j]
        if gap_hi > gap_lo:
            mid = (gap_lo + gap_hi) / 2
            dist = abs(close_val - mid) / (close_val + 1e-8)
            if dist < best_dist:
                best_dist = dist
                found = 1.0
        # Bearish FVG
        gap_hi2 = l[-j-1]
        gap_lo2 = h[-j+1] if -j+1 < 0 else h[len(h)-j]
        if gap_hi2 < gap_lo2:
            mid = (gap_hi2 + gap_lo2) / 2
            dist = abs(close_val - mid) / (close_val + 1e-8)
            if dist < best_dist:
                best_dist = dist
                found = 1.0

    dist_norm = float(np.clip(best_dist * 100, 0, 5) / 5)
    return found, dist_norm


class CryptoTradingEnv(gym.Env):
    """
    Crypto RL Environment — 24/7, price-relative, Binance-compatible.
    Works for BTC/USDT, ETH/USDT, SOL/USDT etc.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_capital: float = 1000.0,
                 trade_pct: float = 0.95,      # % of capital per trade
                 fee_pct: float = 0.0004,        # 0.04% Binance maker fee
                 sl_pct: float = 0.004,          # 0.4% stop loss
                 tp_pct: float = 0.005,          # 0.5% take profit
                 max_hold_bars: int = 96):        # 8 hours max hold

        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.trade_pct = trade_pct
        self.fee_pct = fee_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.max_hold_bars = max_hold_bars

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(27,), dtype=np.float32
        )

        self.start_idx = 100
        self.current_idx = self.start_idx
        self._reset_state()

    def _reset_state(self):
        self.capital = self.initial_capital
        self.position = 0       # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0

    def _get_obs(self, i):
        df = self.df
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)

        c = close.iloc[i]

        # [0-5] Multi-period returns
        rets = []
        for n in [1, 3, 6, 12, 24, 48]:
            if i >= n:
                r = (c - close.iloc[i-n]) / (close.iloc[i-n] + 1e-8)
            else:
                r = 0.0
            rets.append(float(np.clip(r * 10, -1, 1)))

        # [6-8] Candle structure
        o = df['open'].iloc[i] if 'open' in df.columns else c
        body = abs(c - o) / (abs(high.iloc[i] - low.iloc[i]) + 1e-8)
        uw = (high.iloc[i] - max(c, o)) / (abs(high.iloc[i] - low.iloc[i]) + 1e-8)
        lw = (min(c, o) - low.iloc[i]) / (abs(high.iloc[i] - low.iloc[i]) + 1e-8)
        candle = [float(np.clip(body, 0, 1)),
                  float(np.clip(uw, 0, 1)),
                  float(np.clip(lw, 0, 1))]

        # [9-11] HTF bias
        h4, daily, weekly = _calc_htf_bias(close, i)

        # [12-13] FVG
        fvg_present, fvg_dist = _find_fvg(high, low, c, i, high, low)

        # [14] OB proximity (simplified: distance to recent swing)
        lookback = min(20, i)
        swing_hi = high.iloc[i-lookback:i].max()
        swing_lo = low.iloc[i-lookback:i].min()
        ob_prox = float(np.clip((c - swing_lo) / (swing_hi - swing_lo + 1e-8) * 2 - 1, -1, 1))

        # [15-17] Momentum indicators
        s = close.iloc[max(0, i-30):i+1]
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        rsi_val = gain / (gain + loss + 1e-8)
        rsi_norm = float(np.clip(rsi_val * 2 - 1, -1, 1))

        momentum_6 = float(np.clip((c - close.iloc[max(0,i-6)]) / (close.iloc[max(0,i-6)] + 1e-8) * 50, -1, 1))

        vol_20 = close.iloc[max(0,i-20):i+1].pct_change().std()
        volatility = float(np.clip(vol_20 * 100, 0, 1))

        # [18-20] Volume features
        v = vol.iloc[i]
        v_avg = vol.iloc[max(0,i-20):i].mean() + 1e-8
        vol_ratio = float(np.clip(v / v_avg - 1, -1, 1))
        vol_trend = float(np.clip(
            (vol.iloc[max(0,i-5):i].mean() - vol.iloc[max(0,i-20):i-5].mean()) / (v_avg), -1, 1
        ))
        # Buy pressure: close position in high-low range
        buy_pressure = float(np.clip(
            (c - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + 1e-8) * 2 - 1, -1, 1
        ))

        # [21-23] Position state
        in_trade = 1.0 if self.position != 0 else 0.0
        direction = float(self.position)  # -1, 0, 1
        if self.position != 0 and self.entry_price > 0:
            upnl_pct = (c - self.entry_price) / self.entry_price * self.position
            upnl_norm = float(np.clip(upnl_pct * 20, -1, 1))
        else:
            upnl_norm = 0.0

        # [24-25] Bars held, drawdown
        bars_held_norm = float(np.clip(self.bars_held / self.max_hold_bars, 0, 1))
        dd_norm = float(np.clip(self.max_drawdown * 10, 0, 1))

        # [26] Time encoding
        if 'Datetime' in df.columns:
            try:
                hour = pd.Timestamp(df['Datetime'].iloc[i]).hour
                hour_sin = float(np.sin(2 * np.pi * hour / 24))
            except:
                hour_sin = 0.0
        else:
            hour_sin = 0.0

        obs = np.array(
            rets +                    # 6
            candle +                  # 3
            [h4, daily, weekly] +     # 3
            [fvg_present, fvg_dist] + # 2
            [ob_prox] +               # 1
            [rsi_norm, momentum_6, volatility] +  # 3
            [vol_ratio, vol_trend, buy_pressure] + # 3
            [in_trade, direction, upnl_norm] +     # 3
            [bars_held_norm, dd_norm] +            # 2
            [hour_sin],                            # 1
            dtype=np.float32
        )
        assert len(obs) == 27, f"Expected 27 features, got {len(obs)}"
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self._reset_state()
        return self._get_obs(self.current_idx), {}

    def step(self, action):
        df = self.df
        close = df['close']
        high  = df['high']
        low   = df['low']

        reward = -0.0001  # tiny per-bar cost — breaks do-nothing trap
        info   = {}
        done   = False

        c = close.iloc[self.current_idx]

        # ── Entry logic ──────────────────────────────────────────────
        if self.position == 0:
            if action == 1:  # BUY long
                self.position = 1
                self.entry_price = c * (1 + self.fee_pct)
                self.bars_held = 0
                self.total_trades += 1
            elif action == 2:  # SELL short — disabled for spot trading (no-op)
                pass

        # ── In-trade management ──────────────────────────────────────
        elif self.position != 0:
            self.bars_held += 1

            # Unrealized PnL %
            if self.position == 1:
                upnl_pct = (c - self.entry_price) / self.entry_price
            else:
                upnl_pct = (self.entry_price - c) / self.entry_price

            # Per-bar reward = change in unrealized PnL
            prev_upnl = self.unrealized_pnl
            self.unrealized_pnl = upnl_pct * self.capital * self.trade_pct
            reward += (self.unrealized_pnl - prev_upnl) * 0.1

            # Drawdown penalty
            cur_equity = self.capital + self.unrealized_pnl
            dd = (self.peak_capital - cur_equity) / (self.peak_capital + 1e-8)
            if dd > 0.03:
                reward -= 1.0

            # Check SL/TP — TP checked first (wider, more favorable)
            close_reason = None
            if self.position == 1:
                if high.iloc[self.current_idx] >= self.entry_price * (1 + self.tp_pct):
                    close_reason = 'TP'
                    c_exit = self.entry_price * (1 + self.tp_pct)
                elif low.iloc[self.current_idx] <= self.entry_price * (1 - self.sl_pct):
                    close_reason = 'SL'
                    c_exit = self.entry_price * (1 - self.sl_pct)
            else:
                if low.iloc[self.current_idx] <= self.entry_price * (1 - self.tp_pct):
                    close_reason = 'TP'
                    c_exit = self.entry_price * (1 - self.tp_pct)
                elif high.iloc[self.current_idx] >= self.entry_price * (1 + self.sl_pct):
                    close_reason = 'SL'
                    c_exit = self.entry_price * (1 + self.sl_pct)

            # Agent close action
            if action == 3 and close_reason is None:
                close_reason = 'AGENT'
                c_exit = c * (1 - self.fee_pct if self.position == 1 else 1 + self.fee_pct)

            # Forced close
            if self.bars_held >= self.max_hold_bars and close_reason is None:
                close_reason = 'TIMEOUT'
                c_exit = c
                reward -= 5.0

            # Execute close
            if close_reason:
                if self.position == 1:
                    pnl_pct = (c_exit - self.entry_price) / self.entry_price
                else:
                    pnl_pct = (self.entry_price - c_exit) / self.entry_price

                pnl_dollar = pnl_pct * self.capital * self.trade_pct
                self.capital += pnl_dollar
                reward += pnl_pct * 20  # realized PnL reward

                if pnl_pct > 0:
                    self.wins += 1
                else:
                    self.losses += 1

                # Update peak + drawdown
                if self.capital > self.peak_capital:
                    self.peak_capital = self.capital
                dd = (self.peak_capital - self.capital) / (self.peak_capital + 1e-8)
                self.max_drawdown = max(self.max_drawdown, dd)

                self.position = 0
                self.entry_price = 0.0
                self.unrealized_pnl = 0.0
                self.bars_held = 0

        # ── Advance bar ──────────────────────────────────────────────
        self.current_idx += 1
        if self.current_idx >= len(df) - 1:
            done = True
            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
            reward += pnl_pct * 10 + wr * 2
            info = {
                "total_pnl": self.capital - self.initial_capital,
                "total_pnl_pct": pnl_pct,
                "win_rate": wr,
                "wins": self.wins,
                "losses": self.losses,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
            }

        obs = self._get_obs(min(self.current_idx, len(df) - 1))
        return obs, reward, done, False, info

    def render(self):
        wr = self.wins / max(1, self.wins + self.losses)
        pos = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position]
        print(f"Bar {self.current_idx} | {pos} | "
              f"Capital: ${self.capital:.2f} | "
              f"PnL: ${self.capital - self.initial_capital:+.2f} | "
              f"WR: {wr:.1%} ({self.wins}W/{self.losses}L) | "
              f"Trades: {self.total_trades}")


if __name__ == "__main__":
    from binance.client import Client
    import os

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_SECRET_KEY", "")

    print("📥 Fetching BTC/USDT 5m bars from Binance testnet...")
    # Use mainnet public API for historical data (no auth needed)
    # Testnet has no real historical data — mainnet is correct for backtesting
    client = Client("", "", tld='com')

    klines = client.get_historical_klines(
        "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
        "60 days ago UTC"
    )

    df = pd.DataFrame(klines, columns=[
        'Datetime','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df[['Datetime','open','high','low','close','volume']].reset_index(drop=True)

    print(f"✅ {len(df)} bars loaded ({df['Datetime'].iloc[0].date()} → {df['Datetime'].iloc[-1].date()})")

    print("\n🏗️  Building crypto environment...")
    env = CryptoTradingEnv(df)
    obs, _ = env.reset()

    print(f"✅ Environment created")
    print(f"   Observation: {obs.shape} features")
    print(f"   Actions: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE")
    print(f"   Capital: $1,000 USDT | Fee: 0.1% | SL: 1.5% | TP: 2.0%")

    print("\n🎲 Random agent baseline...")
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\n📊 Random Agent Results:")
    print(f"   PnL:      ${info.get('total_pnl', 0):+.2f} ({info.get('total_pnl_pct', 0):.1%})")
    print(f"   WR:       {info.get('win_rate', 0):.1%}")
    print(f"   Trades:   {info.get('total_trades', 0)}")
    print(f"   Max DD:   {info.get('max_drawdown', 0):.1%}")
    print(f"\n✅ Crypto environment ready")
