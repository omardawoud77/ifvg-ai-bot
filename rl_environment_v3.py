"""
NQ Futures RL Environment v3 — Fully Autonomous
================================================
No hardcoded ICT gates. Agent sees everything, decides everything.

Key differences from v2:
- NO entry restrictions — agent can enter any bar
- ICT features are INFORMATION, not walls
- Agent controls EXIT too (action 3 = CLOSE)
- Reward = real PnL per bar while in trade (not just win/loss)
- Agent learns from bad trades — punishment teaches ICT organically
- Max hold = 48 bars (4 hours) before forced close

Observation (32 features):
  [0-7]   Price structure: returns, ATR, candle body, high/low position
  [8-11]  HTF bias: weekly, daily, 4H, 1H (as continuous values, not gates)
  [12-15] ICT structure: FVG present+dist, OB present+dist
  [16-17] BOS + liquidity sweep signal
  [18-21] Session encoding: london, ny_open, ny_pm, london_close (one-hot)
  [22-25] RSI, volume ratio, momentum, volatility
  [26-29] Position state: in_trade, direction, unrealized_pnl, bars_held
  [30-31] Time: hour_sin, hour_cos

Actions:
  0 = HOLD / STAY
  1 = BUY (enter long, or ignored if already in trade)
  2 = SELL (enter short, or ignored if already in trade)
  3 = CLOSE (exit current trade, or ignored if flat)

Reward:
  - Per bar while in trade: unrealized PnL change (real P&L feedback)
  - On close: realized PnL
  - Drawdown penalty: -0.5 per bar if drawdown > 5%
  - Forced close penalty: -20 if held > 48 bars (teaches discipline)
  - No entry bonus or penalty — agent decides freely
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _calc_htf_bias(close, i):
    """HTF bias as continuous [-1, 1] values — information, not gates."""
    s = close.iloc[max(0, i-1500):i+1]
    if len(s) < 100:
        return 0.0, 0.0, 0.0, 0.0

    def bias(fast, slow):
        f = _ema(s, fast).iloc[-1]
        sl = _ema(s, slow).iloc[-1]
        diff = (f - sl) / (abs(sl) + 1e-8)
        return float(np.clip(diff * 100, -1, 1))

    weekly = bias(576, 1440)
    daily  = bias(288, 576)
    h4     = bias(48, 96)
    h1     = bias(12, 24)
    return weekly, daily, h4, h1


def _find_fvg(high, low, close_val, i, df_high, df_low):
    """Find nearest FVG in either direction, return (present, distance, bullish)."""
    lookback = min(50, i)
    h = df_high.iloc[i-lookback:i].values
    l = df_low.iloc[i-lookback:i].values
    c = close_val

    best_dist = 999
    best_present = 0.0
    best_dir = 0.0

    for j in range(2, len(h)-1):
        # Bullish FVG
        gap_lo = h[-j-1]
        gap_hi = l[-j+1]
        if gap_hi > gap_lo:
            mid = (gap_lo + gap_hi) / 2
            dist = (c - mid) / (c * 0.001 + 1e-8)
            if abs(dist) < abs(best_dist):
                best_dist = dist
                best_present = 1.0
                best_dir = 1.0

        # Bearish FVG
        gap_hi2 = l[-j-1]
        gap_lo2 = h[-j+1]
        if gap_hi2 < gap_lo2:
            mid = (gap_hi2 + gap_lo2) / 2
            dist = (mid - c) / (c * 0.001 + 1e-8)
            if abs(dist) < abs(best_dist):
                best_dist = dist
                best_present = 1.0
                best_dir = -1.0

    if best_present == 0:
        return 0.0, 0.0
    return best_present, float(np.clip(best_dist, -3, 3) / 3)


def _find_ob(high, low, close_val, i, df_high, df_low):
    """Find nearest OB, return (present, distance)."""
    lookback = min(30, i)
    h = df_high.iloc[i-lookback:i].values
    l = df_low.iloc[i-lookback:i].values
    c = close_val

    for j in range(3, len(h)-1):
        # Bullish OB: down candle before up move
        if l[-j] < l[-j-1] and h[-j+1] > h[-j]:
            ob_hi = h[-j]
            ob_lo = l[-j]
            if ob_lo <= c <= ob_hi * 1.001:
                dist = (c - (ob_lo + ob_hi)/2) / (c * 0.001 + 1e-8)
                return 1.0, float(np.clip(dist, -3, 3) / 3)

        # Bearish OB: up candle before down move
        if h[-j] > h[-j-1] and l[-j+1] < l[-j]:
            ob_hi = h[-j]
            ob_lo = l[-j]
            if ob_lo <= c <= ob_hi * 1.001:
                dist = ((ob_lo + ob_hi)/2 - c) / (c * 0.001 + 1e-8)
                return 1.0, float(np.clip(dist, -3, 3) / 3)

    return 0.0, 0.0


def _calc_bos(high, low, i):
    """Break of structure signal."""
    lookback = min(20, i)
    h = high.iloc[i-lookback:i].values
    l = low.iloc[i-lookback:i].values
    if len(h) < 5:
        return 0.0
    recent_high = max(h[:-3])
    recent_low  = min(l[:-3])
    if h[-1] > recent_high:
        return 1.0   # bullish BOS
    if l[-1] < recent_low:
        return -1.0  # bearish BOS
    return 0.0


def _session_flags(dt):
    """One-hot session encoding."""
    mins = dt.hour * 60 + dt.minute
    london      = 1.0 if 540  <= mins < 810  else 0.0
    ny_open     = 1.0 if 810  <= mins < 900  else 0.0
    london_close= 1.0 if 900  <= mins < 960  else 0.0
    ny_pm       = 1.0 if 1050 <= mins < 1140 else 0.0
    return london, ny_open, london_close, ny_pm


class NQTradingEnvV3(gym.Env):
    """
    Fully autonomous NQ trading environment.
    No ICT gates — agent sees all, decides all.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 point_value: float = 20.0,
                 max_hold_bars: int = 48,
                 commission_pts: float = 0.25):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.max_hold_bars = max_hold_bars
        self.commission_pts = commission_pts

        # Actions: HOLD, BUY, SELL, CLOSE
        self.action_space = spaces.Discrete(4)

        # 32 features
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(32,), dtype=np.float32
        )

        self.start_idx = 300
        self._precompute()

    def _precompute(self):
        df = self.df
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta).clip(lower=0).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        self._rsi = (100 - 100 / (1 + rs)).fillna(50) / 100.0

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        self._atr = tr.rolling(14).mean().fillna(1.0)

        # Volume ratio
        self._vol_ratio = (vol / vol.rolling(20).mean().fillna(1)).fillna(1.0)

        # Momentum (5-bar return)
        self._momentum = close.pct_change(5).fillna(0)

        # Store raw series for per-bar ICT calcs
        self._close = close
        self._high  = high
        self._low   = low

        # Precompute HTF bias for all bars (slow — do once)
        print("   Precomputing HTF bias for all bars...")
        n = len(df)
        self._htf = np.zeros((n, 4), dtype=np.float32)
        for i in range(self.start_idx, n, 12):  # every 12 bars (1hr)
            w, d, h4, h1 = _calc_htf_bias(self._close, i)
            end = min(i + 12, n)
            self._htf[i:end] = [w, d, h4, h1]

        # Precompute datetime for sessions
        if 'Datetime' in df.columns:
            self._dt = pd.to_datetime(df['Datetime']).dt.tz_localize('UTC') if df['Datetime'].dt.tz is None else pd.to_datetime(df['Datetime'])
        elif df.index.name == 'Datetime' or hasattr(df.index, 'hour'):
            self._dt = df.index
        else:
            self._dt = pd.to_datetime(df.iloc[:, 0])

        print("   ✅ Precompute done")

    def _get_obs(self, i):
        c = float(self._close.iloc[i])
        atr = float(self._atr.iloc[i]) + 1e-8

        # [0-7] Price structure
        ret1  = float(self._close.iloc[i] - self._close.iloc[i-1]) / atr
        ret5  = float(self._close.iloc[i] - self._close.iloc[i-5]) / atr
        ret12 = float(self._close.iloc[i] - self._close.iloc[i-12]) / atr
        body  = float(self._close.iloc[i] - self.df['open'].iloc[i]) / atr
        hi_pos = float(self._high.iloc[i] - c) / atr
        lo_pos = float(c - self._low.iloc[i]) / atr
        vol_r  = float(np.clip(self._vol_ratio.iloc[i], 0, 5) / 5)
        mom    = float(np.clip(self._momentum.iloc[i] * 100, -3, 3))

        # [8-11] HTF bias (continuous)
        htf = self._htf[i]

        # [12-13] FVG
        fvg_present, fvg_dist = _find_fvg(
            self._high, self._low, c, i,
            self._high, self._low
        )

        # [14-15] OB
        ob_present, ob_dist = _find_ob(
            self._high, self._low, c, i,
            self._high, self._low
        )

        # [16] BOS
        bos = _calc_bos(self._high, self._low, i)

        # [17] Liquidity sweep
        recent_high = float(self._high.iloc[max(0,i-20):i].max())
        recent_low  = float(self._low.iloc[max(0,i-20):i].min())
        sweep = 0.0
        if float(self._high.iloc[i]) > recent_high and float(self._close.iloc[i]) < recent_high:
            sweep = -1.0  # sweep high = bearish
        elif float(self._low.iloc[i]) < recent_low and float(self._close.iloc[i]) > recent_low:
            sweep = 1.0   # sweep low = bullish

        # [18-21] Session
        try:
            dt = self._dt.iloc[i]
            lon, nyo, lc, nypm = _session_flags(dt)
        except:
            lon, nyo, lc, nypm = 0.0, 0.0, 0.0, 0.0

        # [22-25] Indicators
        rsi  = float(self._rsi.iloc[i]) * 2 - 1  # normalize to [-1,1]
        atr_n = float(np.clip(atr / (c + 1e-8) * 1000, 0, 3))
        vol_n = float(np.clip(self._vol_ratio.iloc[i], 0, 4) / 4) * 2 - 1
        mom_n = float(np.clip(self._momentum.iloc[i] * 50, -1, 1))

        # [26-29] Position state
        in_trade = 1.0 if self.position != 0 else 0.0
        direction = float(self.position)  # 1, -1, or 0
        upnl = float(np.clip(self.unrealized_pnl / (self.initial_capital * 0.01), -3, 3))
        bars_held = float(np.clip(self.bars_held / self.max_hold_bars, 0, 1))

        # [30-31] Time encoding
        try:
            hour = dt.hour + dt.minute / 60
            hour_sin = float(np.sin(2 * np.pi * hour / 24))
            hour_cos = float(np.cos(2 * np.pi * hour / 24))
        except:
            hour_sin, hour_cos = 0.0, 0.0

        obs = np.array([
            np.clip(ret1, -3, 3),
            np.clip(ret5, -3, 3),
            np.clip(ret12, -3, 3),
            np.clip(body, -3, 3),
            np.clip(hi_pos, 0, 3),
            np.clip(lo_pos, 0, 3),
            vol_r,
            mom,
            htf[0], htf[1], htf[2], htf[3],  # weekly, daily, 4H, 1H
            fvg_present, fvg_dist,
            ob_present, ob_dist,
            bos, sweep,
            lon, nyo, lc, nypm,
            rsi, atr_n, vol_n, mom_n,
            in_trade, direction, upnl, bars_held,
            hour_sin, hour_cos,
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.start_idx
        self.capital = self.initial_capital
        self.position = 0       # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        return self._get_obs(self.current_idx), {}

    def step(self, action):
        i = self.current_idx
        close = float(self._close.iloc[i])
        reward = 0.0
        info = {}
        done = False

        # ── Execute action ────────────────────────────────────────────
        if self.position == 0:  # FLAT
            if action == 1:  # BUY
                self.position = 1
                self.entry_price = close + self.commission_pts
                self.bars_held = 0
                self.unrealized_pnl = 0.0
                self.total_trades += 1

            elif action == 2:  # SELL
                self.position = -1
                self.entry_price = close - self.commission_pts
                self.bars_held = 0
                self.unrealized_pnl = 0.0
                self.total_trades += 1

        elif self.position != 0:  # IN TRADE
            # Update unrealized PnL
            if self.position == 1:
                self.unrealized_pnl = (close - self.entry_price) * self.point_value
            else:
                self.unrealized_pnl = (self.entry_price - close) * self.point_value

            self.bars_held += 1

            # Per-bar reward = PnL change (teaches agent to feel trade progress)
            prev_upnl = reward  # will update below
            reward = self.unrealized_pnl * 0.001  # small per-bar signal

            # Drawdown penalty
            current_equity = self.capital + self.unrealized_pnl
            dd = (self.peak_capital - current_equity) / self.peak_capital
            if dd > 0.05:
                reward -= 0.5

            # Close conditions
            should_close = False
            close_reason = ""

            if action == 3:  # Agent chose to CLOSE
                should_close = True
                close_reason = "agent"
            elif self.bars_held >= self.max_hold_bars:  # Force close
                should_close = True
                close_reason = "timeout"
                reward -= 20  # timeout punishment — teaches planning

            if should_close:
                # Realize PnL
                self.capital += self.unrealized_pnl
                realized = self.unrealized_pnl

                if realized > 0:
                    self.wins += 1
                    reward += realized * 0.1  # bonus for profitable close
                else:
                    self.losses += 1
                    reward += realized * 0.1  # penalty for loss close

                # Update peak / drawdown
                if self.capital > self.peak_capital:
                    self.peak_capital = self.capital
                dd = (self.peak_capital - self.capital) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, dd)

                self.position = 0
                self.entry_price = 0.0
                self.bars_held = 0
                self.unrealized_pnl = 0.0

        # ── Advance bar ───────────────────────────────────────────────
        self.current_idx += 1

        if self.current_idx >= len(self.df) - 1:
            done = True
            # Force close any open trade
            if self.position != 0:
                close_final = float(self._close.iloc[self.current_idx - 1])
                if self.position == 1:
                    self.unrealized_pnl = (close_final - self.entry_price) * self.point_value
                else:
                    self.unrealized_pnl = (self.entry_price - close_final) * self.point_value
                self.capital += self.unrealized_pnl
                self.position = 0

            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
            reward += pnl_pct * 10 + wr * 2  # terminal bonus

            info = {
                "total_pnl": self.capital - self.initial_capital,
                "win_rate": wr,
                "wins": self.wins,
                "losses": self.losses,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
            }

        obs = self._get_obs(min(self.current_idx, len(self.df) - 1))
        return obs, reward, done, False, info

    def render(self):
        wr = self.wins / max(1, self.wins + self.losses)
        pos = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position]
        print(f"Bar {self.current_idx} | {pos} | "
              f"Capital: ${self.capital:.0f} | "
              f"UPNL: ${self.unrealized_pnl:.0f} | "
              f"WR: {wr:.1%} ({self.wins}W/{self.losses}L) | "
              f"Trades: {self.total_trades}")


if __name__ == "__main__":
    print("📥 Loading data...")
    df = pd.read_pickle("nq_archive.pkl")
    df = df.reset_index()
    print(f"✅ {len(df)} bars")

    print("\n🏗️  Building v3 environment (no restrictions)...")
    env = NQTradingEnvV3(df)
    obs, _ = env.reset()

    print(f"\n✅ Environment v3 created")
    print(f"   Observation: {obs.shape} features")
    print(f"   Actions: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE")
    print(f"   No ICT gates — agent decides everything")

    print("\n🎲 Random agent baseline...")
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"\n📊 Random Agent v3:")
    print(f"   PnL:      ${info.get('total_pnl', 0):+,.0f}")
    print(f"   WR:       {info.get('win_rate', 0):.1%}")
    print(f"   Trades:   {info.get('total_trades', 0)}")
    print(f"   Max DD:   {info.get('max_drawdown', 0):.1%}")
    print(f"\n✅ v3 ready — run rl_train_v3.py to train")
