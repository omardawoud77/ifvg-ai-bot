"""
NQ Futures RL Environment v2 — ICT-Aware
==========================================
Upgraded from v1 with full ICT feature set:

Observation space: 28 features (was 15)
New features added:
  - HTF alignment (weekly/daily/4H/1H bias)
  - FVG detection + distance
  - Order Block proximity
  - BOS (Break of Structure)
  - Liquidity sweep signal
  - Kill zone (proper UTC session gates)
  - Swing high/low distance
  - Position self-awareness (PnL, bars held)

The agent now sees what your ICT bot sees — and learns
to optimize within that same universe.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _calc_htf_bias(close, i):
    """
    Calculate multi-timeframe bias at bar i.
    Uses EMA crossovers as proxy for weekly/daily/4H/1H bias.
    5m bars: 1H=12, 4H=48, Daily=288, Weekly=1440
    """
    s = close.iloc[max(0, i-600):i+1]
    if len(s) < 50:
        return 0, 0, 0, 0

    weekly = 1 if _ema(s, 576).iloc[-1] > _ema(s, 1440).iloc[-1] else -1
    daily  = 1 if _ema(s, 288).iloc[-1] > _ema(s, 576).iloc[-1]  else -1
    h4     = 1 if _ema(s, 48).iloc[-1]  > _ema(s, 96).iloc[-1]   else -1
    h1     = 1 if _ema(s, 12).iloc[-1]  > _ema(s, 24).iloc[-1]   else -1
    return weekly, daily, h4, h1


def _find_fvg(high, low, close, i, direction):
    """Detect FVG and return normalized distance to it."""
    sl = slice(max(0, i-30), i+1)
    h = high.iloc[sl].values
    l = low.iloc[sl].values
    c = close.iloc[i]

    for j in range(2, min(30, len(h)-1)):
        if direction == 1:  # long — bullish FVG
            gap_lo = h[-j-1]
            gap_hi = l[-j+1]
            if gap_hi > gap_lo:
                mid = (gap_lo + gap_hi) / 2
                dist = (c - mid) / (c * 0.001 + 1e-8)
                return 1.0, float(np.clip(dist, -5, 5) / 5)
        else:  # short — bearish FVG
            gap_hi = l[-j-1]
            gap_lo = h[-j+1]
            if gap_hi < gap_lo:
                mid = (gap_lo + gap_hi) / 2
                dist = (mid - c) / (c * 0.001 + 1e-8)
                return 1.0, float(np.clip(dist, -5, 5) / 5)
    return 0.0, 0.0


def _find_ob(high, low, close, i, direction):
    """Detect Order Block and return proximity."""
    sl = slice(max(0, i-30), i+1)
    h = high.iloc[sl].values
    l = low.iloc[sl].values
    c = close.iloc[i]
    avg_range = np.mean(h - l)

    for j in range(3, min(30, len(h))):
        bar_range = h[-j] - l[-j]
        if bar_range > avg_range * 1.5:
            ob_mid = (h[-j] + l[-j]) / 2
            dist = abs(c - ob_mid) / (avg_range + 1e-8)
            at_ob = 1.0 if dist < 2.0 else 0.0
            return at_ob, float(np.clip(dist / 5, 0, 1) * 2 - 1)
    return 0.0, 0.0


def _detect_bos(high, low, close, i, direction):
    """Break of Structure — recent swing broken."""
    sl = slice(max(0, i-20), i+1)
    h = high.iloc[sl].values
    l = low.iloc[sl].values
    c = float(close.iloc[i])

    if direction == 1:  # long
        recent_high = np.max(h[:-3]) if len(h) > 3 else h[0]
        return 1.0 if c > recent_high else 0.0
    else:  # short
        recent_low = np.min(l[:-3]) if len(l) > 3 else l[0]
        return 1.0 if c < recent_low else 0.0


def _kill_zone(dt):
    """Returns (session_id 0-4, is_active) for UTC datetime."""
    mins = dt.hour * 60 + dt.minute
    if 540 <= mins < 810:   return 1, 1.0   # london main
    if 810 <= mins < 900:   return 2, 1.0   # ny open
    if 900 <= mins < 960:   return 3, 1.0   # london close
    if 1050 <= mins < 1140: return 4, 1.0   # ny pm
    return 0, 0.0                            # off-hours


class NQTradingEnvV2(gym.Env):
    """
    ICT-aware NQ trading environment.
    28-feature observation space including HTF, FVG, OB, BOS, sessions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 sl_pts: float = 30.0, tp_pts: float = 30.0,
                 cooldown_bars: int = 6, max_daily_trades: int = 5):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.sl_pts  = sl_pts
        self.tp_pts  = tp_pts
        self.cooldown_bars    = cooldown_bars
        self.max_daily_trades = max_daily_trades

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(28,), dtype=np.float32
        )

        self.start_idx = 600  # need 600 bars for weekly EMA
        self._precompute_base_features()
        self._precompute_ict_features()

    def _precompute_base_features(self):
        df    = self.df
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        ema9   = _ema(close, 9)
        ema21  = _ema(close, 21)
        ema50  = _ema(close, 50)
        ema200 = _ema(close, 200)

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, 0.001))

        tr  = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        vol_avg   = vol.rolling(20).mean()
        vol_ratio = (vol / vol_avg.replace(0, 1)).clip(0, 5)

        hi20 = high.rolling(20).max()
        lo20 = low.rolling(20).min()
        hi50 = high.rolling(50).max()
        lo50 = low.rolling(50).min()

        price_pos20 = (close - lo20) / (hi20 - lo20 + 1e-8)
        price_pos50 = (close - lo50) / (hi50 - lo50 + 1e-8)

        body = (close - df['open']) / (atr + 1e-8)

        self._base = {
            'ema9': ema9, 'ema21': ema21, 'ema50': ema50, 'ema200': ema200,
            'rsi': rsi, 'atr': atr, 'vol_ratio': vol_ratio,
            'price_pos20': price_pos20, 'price_pos50': price_pos50,
            'body': body, 'close': close, 'high': high, 'low': low,
        }

    def _precompute_ict_features(self):
        """Precompute all ICT features bar by bar."""
        n = len(self.df)
        close = self._base['close']
        high  = self._base['high']
        low   = self._base['low']
        atr   = self._base['atr']

        # Pre-allocate arrays
        self._htf_weekly = np.zeros(n)
        self._htf_daily  = np.zeros(n)
        self._htf_h4     = np.zeros(n)
        self._htf_h1     = np.zeros(n)
        self._htf_align  = np.zeros(n)
        self._fvg_present = np.zeros(n)
        self._fvg_dist   = np.zeros(n)
        self._ob_present = np.zeros(n)
        self._ob_dist    = np.zeros(n)
        self._bos        = np.zeros(n)
        self._kz_active  = np.zeros(n)
        self._kz_session = np.zeros(n)

        # Parse datetimes
        dt_col = 'Datetime' if 'Datetime' in self.df.columns else 'datetime'
        has_dt = dt_col in self.df.columns

        print(f"  Precomputing ICT features for {n} bars...")
        for i in range(self.start_idx, n):
            # HTF bias (computed every 12 bars = 1H to save time)
            if i % 12 == 0:
                w, d, h4, h1 = _calc_htf_bias(close, i)
                self._htf_weekly[i] = w
                self._htf_daily[i]  = d
                self._htf_h4[i]     = h4
                self._htf_h1[i]     = h1
            else:
                self._htf_weekly[i] = self._htf_weekly[i-1]
                self._htf_daily[i]  = self._htf_daily[i-1]
                self._htf_h4[i]     = self._htf_h4[i-1]
                self._htf_h1[i]     = self._htf_h1[i-1]

            direction = 1 if self._htf_h4[i] > 0 else -1
            align = (self._htf_weekly[i] == self._htf_daily[i] == self._htf_h4[i])
            self._htf_align[i] = 1.0 if align else 0.0

            # FVG
            fp, fd = _find_fvg(high, low, close, i, direction)
            self._fvg_present[i] = fp
            self._fvg_dist[i]    = fd

            # OB
            op, od = _find_ob(high, low, close, i, direction)
            self._ob_present[i] = op
            self._ob_dist[i]    = od

            # BOS
            self._bos[i] = _detect_bos(high, low, close, i, direction)

            # Kill zone
            if has_dt:
                try:
                    dt = pd.Timestamp(self.df[dt_col].iloc[i])
                    sess_id, kz = _kill_zone(dt)
                    self._kz_active[i]  = kz
                    self._kz_session[i] = sess_id / 4.0  # normalize 0-1
                except:
                    pass

        print(f"  ✅ ICT features computed")

    def _get_obs(self):
        i = self.current_idx
        b = self._base
        c = float(b['close'].iloc[i])
        atr = float(b['atr'].iloc[i]) + 1e-8

        direction = 1 if self._htf_h4[i] > 0 else -1

        obs = np.array([
            # ── Base technical (8) ──────────────────────────────────────
            float(np.clip((b['ema9'].iloc[i] - b['ema21'].iloc[i]) / atr / 3, -1, 1)),
            float(np.clip((b['ema21'].iloc[i] - b['ema50'].iloc[i]) / atr / 3, -1, 1)),
            float(np.clip((b['ema50'].iloc[i] - b['ema200'].iloc[i]) / atr / 3, -1, 1)),
            float(np.clip(b['rsi'].iloc[i] / 50 - 1, -1, 1)),
            float(np.clip(b['vol_ratio'].iloc[i] / 2.5 - 1, -1, 1)),
            float(b['price_pos20'].iloc[i] * 2 - 1),
            float(b['price_pos50'].iloc[i] * 2 - 1),
            float(np.clip(b['body'].iloc[i] / 2, -1, 1)),
            # ── HTF alignment (5) ───────────────────────────────────────
            float(self._htf_weekly[i]),      # +1 bull / -1 bear
            float(self._htf_daily[i]),
            float(self._htf_h4[i]),
            float(self._htf_h1[i]),
            float(self._htf_align[i] * 2 - 1),  # +1 aligned / -1 not
            # ── ICT structure (5) ───────────────────────────────────────
            float(self._fvg_present[i] * 2 - 1),
            float(self._fvg_dist[i]),
            float(self._ob_present[i] * 2 - 1),
            float(self._ob_dist[i]),
            float(self._bos[i] * 2 - 1),
            # ── Session (3) ─────────────────────────────────────────────
            float(self._kz_active[i] * 2 - 1),
            float(self._kz_session[i] * 2 - 1),
            float(direction),                # ICT direction bias
            # ── Returns (3) ─────────────────────────────────────────────
            float(np.clip(b['close'].pct_change(3).iloc[i] / 0.02, -1, 1)),
            float(np.clip(b['close'].pct_change(12).iloc[i] / 0.05, -1, 1)),
            float(np.clip(b['close'].pct_change(48).iloc[i] / 0.10, -1, 1)),
            # ── Position state (4) ──────────────────────────────────────
            float(self.position),            # -1/0/1
            float(np.clip((c - self.entry_price) / atr / 3, -1, 1)) if self.position != 0 else 0.0,
            float(np.clip(self.bars_in_trade / 30, 0, 1) * 2 - 1),
            float(np.clip(self.max_drawdown * 10, 0, 1) * 2 - 1),
        ], dtype=np.float32)

        return np.nan_to_num(obs, 0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx    = self.start_idx
        self.capital        = self.initial_capital
        self.position       = 0
        self.entry_price    = 0.0
        self.last_trade_bar = -999
        self.daily_trades   = {}
        self.wins           = 0
        self.losses         = 0
        self.max_capital    = self.initial_capital
        self.max_drawdown   = 0.0
        self.bars_in_trade  = 0
        return self._get_obs(), {}

    def step(self, action):
        bar   = self.df.iloc[self.current_idx]
        close = float(bar['close'])
        reward = 0.0
        info   = {}

        # ── Manage open trade ────────────────────────────────────────────
        if self.position != 0:
            self.bars_in_trade += 1
            if self.position == 1:
                if float(bar['low']) <= self.entry_price - self.sl_pts:
                    reward = -1.0; self.losses += 1
                    self.capital -= self.sl_pts * 2; self.position = 0; self.bars_in_trade = 0
                elif float(bar['high']) >= self.entry_price + self.tp_pts:
                    reward = +1.0; self.wins += 1
                    self.capital += self.tp_pts * 2; self.position = 0; self.bars_in_trade = 0
            else:
                if float(bar['high']) >= self.entry_price + self.sl_pts:
                    reward = -1.0; self.losses += 1
                    self.capital -= self.sl_pts * 2; self.position = 0; self.bars_in_trade = 0
                elif float(bar['low']) <= self.entry_price - self.tp_pts:
                    reward = +1.0; self.wins += 1
                    self.capital += self.tp_pts * 2; self.position = 0; self.bars_in_trade = 0

        # ── Enter new trade ──────────────────────────────────────────────
        elif action in (1, 2):
            cooldown_ok = (self.current_idx - self.last_trade_bar) >= self.cooldown_bars
            day = str(self.current_idx // 78)
            daily_ok = self.daily_trades.get(day, 0) < self.max_daily_trades

            # ICT gate: only trade during kill zones with HTF alignment
            ict_ok = self._kz_active[self.current_idx] > 0 and self._htf_align[self.current_idx] > 0

            if cooldown_ok and daily_ok and ict_ok:
                direction = 1 if self._htf_h4[self.current_idx] > 0 else -1
                # Agent must agree with HTF direction
                if (action == 1 and direction == 1) or (action == 2 and direction == -1):
                    self.position    = direction
                    self.entry_price = close
                    self.last_trade_bar = self.current_idx
                    self.daily_trades[day] = self.daily_trades.get(day, 0) + 1
                    self.bars_in_trade = 0
                    reward = +0.05  # small bonus for entering — fights do-nothing trap

        # ── Drawdown tracking ────────────────────────────────────────────
        self.max_capital  = max(self.max_capital, self.capital)
        drawdown = (self.max_capital - self.capital) / self.max_capital
        self.max_drawdown = max(self.max_drawdown, drawdown)
        if drawdown > 0.1:
            reward -= 0.5

        self.current_idx += 1
        done = self.current_idx >= len(self.df) - 1

        if done:
            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
            reward += pnl_pct * 10 + wr * 2
            info = {
                "total_pnl": self.capital - self.initial_capital,
                "win_rate": wr, "wins": self.wins, "losses": self.losses,
                "max_drawdown": self.max_drawdown,
            }

        obs = self._get_obs() if not done else self._get_obs()
        return obs, reward, done, False, info

    def render(self):
        wr = self.wins / max(1, self.wins + self.losses)
        print(f"Bar {self.current_idx} | "
              f"Capital: ${self.capital:.0f} | "
              f"Pos: {'LONG' if self.position==1 else 'SHORT' if self.position==-1 else 'FLAT'} | "
              f"HTF: {'✅' if self._htf_align[self.current_idx] else '❌'} | "
              f"KZ: {'✅' if self._kz_active[self.current_idx] else '❌'} | "
              f"WR: {wr:.1%} ({self.wins}W/{self.losses}L)")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📥 Loading 60d NQ data...")
    df = pd.read_pickle("nq_60d.pkl")
    df = df.reset_index()
    print(f"✅ {len(df)} bars")

    print("\n🏗️  Building v2 environment...")
    env = NQTradingEnvV2(df)
    obs, _ = env.reset()

    print(f"\n✅ Environment v2 created")
    print(f"   Observation shape: {obs.shape}  (was 15, now 28)")
    print(f"   New ICT features: HTF bias, FVG, OB, BOS, kill zones, position state")
    print(f"   Sample obs: {obs[:8].round(3)}")

    print("\n🎲 Random agent test...")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"\n📊 Random Agent v2 Results:")
    print(f"   Steps: {steps}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final PnL: ${info.get('total_pnl', 0):.0f}")
    print(f"   Win rate: {info.get('win_rate', 0):.1%}")
    print(f"   Wins: {info.get('wins', 0)} | Losses: {info.get('losses', 0)}")
    print(f"   Max drawdown: {info.get('max_drawdown', 0):.1%}")
    print(f"\n✅ Environment v2 ready for training")
