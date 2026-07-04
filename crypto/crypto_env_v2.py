"""
Crypto RL Environment — Clean MTF, Zero Hardcoded Logic
========================================================
Agent sees raw OHLCV from 1h, 4h, 1d, 1w timeframes.
No RSI. No FVG. No EMA bias. No indicators.
The neural network discovers patterns on its own.

Observation (36 features) — pure price data only:
  [0-5]   1h: returns (1,2,3,6,12,24 bars)
  [6-8]   1h: candle body, upper wick, lower wick
  [9-10]  1h: volume ratio vs 20-bar avg, buy pressure
  [11-14] 4h: returns (1,2,3,6 bars)
  [15-16] 4h: candle body, volume ratio
  [17-20] 1d: returns (1,2,3,5 bars)
  [21-22] 1d: candle body, volume ratio
  [23-25] 1w: returns (1,2,3 bars)
  [26]    1w: candle body
  [27-30] Position: in_trade, direction, unrealized_pnl_pct, bars_held_norm
  [31-33] Time: hour_sin, hour_cos, day_of_week_sin
  [34-35] Risk: max_drawdown_norm, current_drawdown_norm

Actions:
  0 = HOLD
  1 = BUY (long)
  2 = SELL (short)
  3 = CLOSE

Rules (minimum viable only):
  - Hard SL: 2% — catastrophic loss guard, not strategy
  - Max hold: 48 bars timeout
  - Fee: 0.04% per side
  - HOLD cost: -0.0002/bar — forces agent to seek opportunity
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoMTFEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame,
                 initial_capital: float = 1000.0,
                 trade_pct: float = 0.95,
                 fee_pct: float = 0.0004,
                 sl_pct: float = 0.02,
                 max_hold_bars: int = 48,
                 hold_cost: float = 0.0002):

        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.trade_pct = trade_pct
        self.fee_pct = fee_pct
        self.sl_pct = sl_pct
        self.max_hold_bars = max_hold_bars
        self.hold_cost = hold_cost

        self.N_FEATURES = 36
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N_FEATURES,), dtype=np.float32
        )
        self.start_idx = 30
        self._reset_state()

    def _reset_state(self):
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.current_idx = self.start_idx

    def _ret(self, series, i, n):
        if i < n:
            return 0.0
        r = (series.iloc[i] - series.iloc[i-n]) / (series.iloc[i-n] + 1e-8)
        return float(np.clip(r * 5, -1, 1))

    def _candle(self, o, h, l, c):
        rng = abs(h - l) + 1e-8
        body = float(np.clip(abs(c - o) / rng, 0, 1))
        uw   = float(np.clip((h - max(c, o)) / rng, 0, 1))
        lw   = float(np.clip((min(c, o) - l) / rng, 0, 1))
        return body, uw, lw

    def _vol_ratio(self, vol, i, w=20):
        avg = vol.iloc[max(0, i-w):i].mean() + 1e-8
        return float(np.clip(vol.iloc[i] / avg - 1, -1, 1))

    def _get_obs(self, i):
        df = self.df
        obs = []

        # 1h raw price features
        c = df['close'];  o = df['open'];  h = df['high'];  l = df['low'];  v = df['volume']
        for n in [1,2,3,6,12,24]:
            obs.append(self._ret(c, i, n))
        body, uw, lw = self._candle(o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i])
        obs += [body, uw, lw]
        obs.append(self._vol_ratio(v, i))
        rng = abs(h.iloc[i] - l.iloc[i]) + 1e-8
        obs.append(float(np.clip((c.iloc[i] - l.iloc[i]) / rng * 2 - 1, -1, 1)))

        # 4h raw price features
        c4 = df['h4_close']; o4 = df['h4_open']; h4 = df['h4_high']; l4 = df['h4_low']; v4 = df['h4_volume']
        for n in [1,2,3,6]:
            obs.append(self._ret(c4, i, n))
        body4, _, _ = self._candle(o4.iloc[i], h4.iloc[i], l4.iloc[i], c4.iloc[i])
        obs.append(body4)
        obs.append(self._vol_ratio(v4, i))

        # 1d raw price features
        c1d = df['d1_close']; o1d = df['d1_open']; h1d = df['d1_high']; l1d = df['d1_low']; v1d = df['d1_volume']
        for n in [1,2,3,5]:
            obs.append(self._ret(c1d, i, n))
        body1d, _, _ = self._candle(o1d.iloc[i], h1d.iloc[i], l1d.iloc[i], c1d.iloc[i])
        obs.append(body1d)
        obs.append(self._vol_ratio(v1d, i))

        # 1w raw price features
        c1w = df['w1_close']; o1w = df['w1_open']; h1w = df['w1_high']; l1w = df['w1_low']
        for n in [1,2,3]:
            obs.append(self._ret(c1w, i, n))
        body1w, _, _ = self._candle(o1w.iloc[i], h1w.iloc[i], l1w.iloc[i], c1w.iloc[i])
        obs.append(body1w)

        # Position state
        in_trade = 1.0 if self.position != 0 else 0.0
        direction = float(self.position)
        if self.position != 0 and self.entry_price > 0:
            upnl = (c.iloc[i] - self.entry_price) / self.entry_price * self.position
            upnl_norm = float(np.clip(upnl * 10, -1, 1))
        else:
            upnl_norm = 0.0
        bars_norm = float(np.clip(self.bars_held / self.max_hold_bars, 0, 1))
        obs += [in_trade, direction, upnl_norm, bars_norm]

        # Time encoding
        try:
            ts = pd.Timestamp(df['Datetime'].iloc[i])
            obs += [
                float(np.sin(2 * np.pi * ts.hour / 24)),
                float(np.cos(2 * np.pi * ts.hour / 24)),
                float(np.sin(2 * np.pi * ts.dayofweek / 7))
            ]
        except:
            obs += [0.0, 0.0, 0.0]

        # Risk state
        cur_eq = self.capital + self.unrealized_pnl
        cur_dd = (self.peak_capital - cur_eq) / (self.peak_capital + 1e-8)
        obs += [
            float(np.clip(self.max_drawdown * 5, 0, 1)),
            float(np.clip(cur_dd * 5, 0, 1))
        ]

        assert len(obs) == self.N_FEATURES, f"Expected {self.N_FEATURES}, got {len(obs)}"
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(self.current_idx), {}

    def step(self, action):
        df = self.df
        c = df['close'].iloc[self.current_idx]
        h = df['high'].iloc[self.current_idx]
        l = df['low'].iloc[self.current_idx]

        reward = -self.hold_cost
        info = {}
        done = False

        # Entry
        if self.position == 0:
            if action == 1:
                self.position = 1
                self.entry_price = c * (1 + self.fee_pct)
                self.bars_held = 0
                self.total_trades += 1
            elif action == 2:
                self.position = -1
                self.entry_price = c * (1 - self.fee_pct)
                self.bars_held = 0
                self.total_trades += 1

        # In trade
        elif self.position != 0:
            self.bars_held += 1

            upnl_pct = (c - self.entry_price) / self.entry_price * self.position
            prev_upnl = self.unrealized_pnl
            self.unrealized_pnl = upnl_pct * self.capital * self.trade_pct
            reward += (self.unrealized_pnl - prev_upnl) / self.capital

            # Hard SL only — agent decides everything else
            close_reason = None
            if self.position == 1 and l <= self.entry_price * (1 - self.sl_pct):
                close_reason = 'SL'
                c_exit = self.entry_price * (1 - self.sl_pct)
            elif self.position == -1 and h >= self.entry_price * (1 + self.sl_pct):
                close_reason = 'SL'
                c_exit = self.entry_price * (1 + self.sl_pct)

            if action == 3 and close_reason is None:
                close_reason = 'AGENT'
                c_exit = c * (1 - self.fee_pct if self.position == 1 else 1 + self.fee_pct)

            if self.bars_held >= self.max_hold_bars and close_reason is None:
                close_reason = 'TIMEOUT'
                c_exit = c
                reward -= 0.01

            if close_reason:
                pnl_pct = (c_exit - self.entry_price) / self.entry_price * self.position
                pnl_dollar = pnl_pct * self.capital * self.trade_pct
                self.capital += pnl_dollar
                reward += pnl_pct * 5

                if pnl_pct > 0:
                    self.wins += 1
                else:
                    self.losses += 1

                self.peak_capital = max(self.peak_capital, self.capital)
                dd = (self.peak_capital - self.capital) / (self.peak_capital + 1e-8)
                self.max_drawdown = max(self.max_drawdown, dd)

                # Expose trade-close details for reward-override subclasses
                info['trade_closed'] = True
                info['pnl_pct'] = pnl_pct
                info['close_reason'] = close_reason

                self.position = 0
                self.entry_price = 0.0
                self.unrealized_pnl = 0.0
                self.bars_held = 0

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
              f"WR: {wr:.1%} ({self.wins}W/{self.losses}L)")


# ── Technical Indicator Environment (44 features) ────────────────────────────
# MTF_15M_RETRAIN

class CryptoTechEnv(CryptoMTFEnv):  # MTF_15M_RETRAIN
    """Extends CryptoMTFEnv with 8 technical indicator features (36 → 44).  # MTF_15M_RETRAIN
    The PPO model can now learn which technical conditions produce profitable trades,  # MTF_15M_RETRAIN
    aligning its signals with the reasoning engine's filters."""  # MTF_15M_RETRAIN

    def __init__(self, df, **kwargs):  # MTF_15M_RETRAIN
        super().__init__(df, **kwargs)  # MTF_15M_RETRAIN
        self.N_FEATURES = 44  # MTF_15M_RETRAIN
        self.observation_space = spaces.Box(  # MTF_15M_RETRAIN
            low=-1.0, high=1.0,  # MTF_15M_RETRAIN
            shape=(self.N_FEATURES,), dtype=np.float32  # MTF_15M_RETRAIN
        )  # MTF_15M_RETRAIN

    def _get_obs(self, i):  # MTF_15M_RETRAIN
        # Get base 36 features (temporarily set N_FEATURES=36 for parent assert)  # MTF_15M_RETRAIN
        saved = self.N_FEATURES  # MTF_15M_RETRAIN
        self.N_FEATURES = 36  # MTF_15M_RETRAIN
        base_obs = super()._get_obs(i)  # MTF_15M_RETRAIN
        self.N_FEATURES = saved  # MTF_15M_RETRAIN

        df = self.df  # MTF_15M_RETRAIN
        c = float(df['close'].iloc[i])  # MTF_15M_RETRAIN

        # 1. trend_score: price vs HTF closes  # MTF_15M_RETRAIN
        h4c = float(df['h4_close'].iloc[i])  # MTF_15M_RETRAIN
        d1c = float(df['d1_close'].iloc[i])  # MTF_15M_RETRAIN
        w1c = float(df['w1_close'].iloc[i]) if 'w1_close' in df.columns else d1c  # MTF_15M_RETRAIN
        above = sum([c > h4c * 1.001, c > d1c * 1.001, c > w1c * 1.001])  # MTF_15M_RETRAIN
        trend_score = {3: 1.0, 2: 0.5, 1: -0.5, 0: -1.0}[above]  # MTF_15M_RETRAIN

        # 2. momentum_score: 3-bar return  # MTF_15M_RETRAIN
        prev_c = float(df['close'].iloc[max(0, i - 3)])  # MTF_15M_RETRAIN
        mom = (c - prev_c) / (prev_c + 1e-8)  # MTF_15M_RETRAIN
        if mom > 0.015:    momentum_score = 1.0  # MTF_15M_RETRAIN
        elif mom > 0:      momentum_score = 0.5  # MTF_15M_RETRAIN
        elif mom > -0.015: momentum_score = -0.5  # MTF_15M_RETRAIN
        else:              momentum_score = -1.0  # MTF_15M_RETRAIN

        # 3. volume_score  # MTF_15M_RETRAIN
        vol_w = df['volume'].iloc[max(0, i - 20):i]  # MTF_15M_RETRAIN
        avg_vol = float(vol_w.mean()) + 1e-8  # MTF_15M_RETRAIN
        vol_ratio = float(df['volume'].iloc[i]) / avg_vol  # MTF_15M_RETRAIN
        if vol_ratio > 2.0:   volume_score = 1.0  # MTF_15M_RETRAIN
        elif vol_ratio > 1.4: volume_score = 0.5  # MTF_15M_RETRAIN
        elif vol_ratio > 0.7: volume_score = 0.0  # MTF_15M_RETRAIN
        else:                 volume_score = -0.5  # MTF_15M_RETRAIN

        # 4. volatility_score (ATR-based)  # MTF_15M_RETRAIN
        highs = df['high'].values  # MTF_15M_RETRAIN
        lows = df['low'].values  # MTF_15M_RETRAIN
        closes = df['close'].values  # MTF_15M_RETRAIN
        atr_sum = 0.0  # MTF_15M_RETRAIN
        atr_n = 0  # MTF_15M_RETRAIN
        for j in range(max(1, i - 13), i + 1):  # MTF_15M_RETRAIN
            tr = max(highs[j] - lows[j],  # MTF_15M_RETRAIN
                     abs(highs[j] - closes[j - 1]),  # MTF_15M_RETRAIN
                     abs(lows[j] - closes[j - 1]))  # MTF_15M_RETRAIN
            atr_sum += tr  # MTF_15M_RETRAIN
            atr_n += 1  # MTF_15M_RETRAIN
        atr = atr_sum / max(atr_n, 1)  # MTF_15M_RETRAIN
        atr_pct = atr / max(c, 0.01)  # MTF_15M_RETRAIN
        if atr_pct > 0.025:   volatility_score = 1.0  # MTF_15M_RETRAIN
        elif atr_pct > 0.010: volatility_score = 0.0  # MTF_15M_RETRAIN
        else:                 volatility_score = -1.0  # MTF_15M_RETRAIN

        # 5. regime_score (derived from above)  # MTF_15M_RETRAIN
        if volatility_score == 1.0:  # MTF_15M_RETRAIN
            regime_score = -1.0  # HIGH_VOLATILITY  # MTF_15M_RETRAIN
        elif trend_score == 1.0 and momentum_score >= 0.5 and volume_score >= 0.0:  # MTF_15M_RETRAIN
            regime_score = 0.5  # TRENDING_BULL  # MTF_15M_RETRAIN
        elif trend_score == -1.0 and momentum_score <= -0.5 and volume_score >= 0.0:  # MTF_15M_RETRAIN
            regime_score = 0.5  # TRENDING_BEAR  # MTF_15M_RETRAIN
        elif abs(trend_score) <= 0.5 and abs(momentum_score) <= 0.5 and volume_score <= 0.0:  # MTF_15M_RETRAIN
            regime_score = 1.0  # RANGING  # MTF_15M_RETRAIN
        else:  # MTF_15M_RETRAIN
            regime_score = -0.5  # LOW_QUALITY  # MTF_15M_RETRAIN

        # 6. atr_pct_norm  # MTF_15M_RETRAIN
        atr_pct_norm = float(np.clip((atr_pct - 0.02) / 0.02, -1, 1))  # MTF_15M_RETRAIN

        # 7. fvg_score (3-bar gap)  # MTF_15M_RETRAIN
        fvg_score = 0.0  # MTF_15M_RETRAIN
        if i >= 2:  # MTF_15M_RETRAIN
            bar2_high = float(df['high'].iloc[i - 2])  # MTF_15M_RETRAIN
            bar2_low = float(df['low'].iloc[i - 2])  # MTF_15M_RETRAIN
            cur_low = float(df['low'].iloc[i])  # MTF_15M_RETRAIN
            cur_high = float(df['high'].iloc[i])  # MTF_15M_RETRAIN
            if bar2_high < cur_low:  fvg_score = 1.0  # bullish  # MTF_15M_RETRAIN
            elif bar2_low > cur_high: fvg_score = -1.0  # bearish  # MTF_15M_RETRAIN

        # 8. wick_score  # MTF_15M_RETRAIN
        o = float(df['open'].iloc[i])  # MTF_15M_RETRAIN
        h = float(df['high'].iloc[i])  # MTF_15M_RETRAIN
        l = float(df['low'].iloc[i])  # MTF_15M_RETRAIN
        rng = h - l + 1e-9  # MTF_15M_RETRAIN
        upper_wick = (h - max(o, c)) / rng  # MTF_15M_RETRAIN
        lower_wick = (min(o, c) - l) / rng  # MTF_15M_RETRAIN
        if lower_wick > 0.6:   wick_score = 1.0  # MTF_15M_RETRAIN
        elif upper_wick > 0.6: wick_score = -1.0  # MTF_15M_RETRAIN
        else:                  wick_score = 0.0  # MTF_15M_RETRAIN

        tech = np.array([trend_score, momentum_score, volume_score,  # MTF_15M_RETRAIN
                         volatility_score, regime_score, atr_pct_norm,  # MTF_15M_RETRAIN
                         fvg_score, wick_score], dtype=np.float32)  # MTF_15M_RETRAIN
        obs = np.concatenate([base_obs, tech])  # MTF_15M_RETRAIN
        assert len(obs) == self.N_FEATURES, f"Expected {self.N_FEATURES}, got {len(obs)}"  # MTF_15M_RETRAIN
        return obs  # MTF_15M_RETRAIN


class CryptoTechRREnv(CryptoTechEnv):  # MTF_15M_RETRAIN
    """CryptoTechEnv with R:R-aware reward (penalizes small wins <2.5%)."""  # MTF_15M_RETRAIN
    MIN_PROFIT_PCT = 0.025  # MTF_15M_RETRAIN: 2.5% matches new SL/TP

    def step(self, action):  # MTF_15M_RETRAIN
        obs, reward, done, truncated, info = super().step(action)  # MTF_15M_RETRAIN
        if info.get('trade_closed', False):  # MTF_15M_RETRAIN
            pnl_pct = info.get('pnl_pct', 0)  # MTF_15M_RETRAIN
            if pnl_pct >= self.MIN_PROFIT_PCT:  # MTF_15M_RETRAIN
                reward = 3.0  # MTF_15M_RETRAIN
            elif pnl_pct >= 0:  # MTF_15M_RETRAIN
                reward = -0.5  # MTF_15M_RETRAIN
            else:  # MTF_15M_RETRAIN
                reward = -2.0  # MTF_15M_RETRAIN
        return obs, reward, done, truncated, info  # MTF_15M_RETRAIN


class CryptoRREnv(CryptoMTFEnv):
    """
    Subclass with RR-aware reward. Forces agent to find 1.5%+ moves.
    Small wins are penalized to prevent scalping behavior.

    Reward on trade close:
      pnl >= 1.5%:  +3.0  (meaningful win)
      0 <= pnl < 1.5%: -0.5  (small win penalized — we want real moves)
      pnl < 0:      -2.0  (loss)

    Non-close bars keep the parent's continuous reward (hold cost +
    unrealized PnL delta) for gradient signal during the hold period.
    """
    MIN_PROFIT_PCT = 0.015  # 1.5% minimum gain to reward positively

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Override reward only when a trade closes
        if info.get('trade_closed', False):
            pnl_pct = info.get('pnl_pct', 0)
            if pnl_pct >= self.MIN_PROFIT_PCT:
                reward = 3.0    # meaningful win
            elif pnl_pct >= 0:
                reward = -0.5   # small win penalized
            else:
                reward = -2.0   # loss

        return obs, reward, done, truncated, info


if __name__ == "__main__":
    import os
    pkl = 'btc_mtf.pkl' if os.path.exists('btc_mtf.pkl') else '../crypto/btc_mtf.pkl'
    print("📥 Loading MTF dataset...")
    df = pd.read_pickle(pkl)
    print(f"✅ {len(df)} bars | {len(df.columns)} columns")

    env = CryptoMTFEnv(df)
    obs, _ = env.reset()
    print(f"✅ Observation: {obs.shape} — pure OHLCV, no indicators")

    print("\n🎲 Random baseline...")
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, done, _, info = env.step(env.action_space.sample())

    print(f"\n📊 Random baseline (1h MTF, clean):")
    print(f"   PnL:    ${info.get('total_pnl',0):+.2f} ({info.get('total_pnl_pct',0):+.1%})")
    print(f"   WR:     {info.get('win_rate',0):.1%}")
    print(f"   Trades: {info.get('total_trades',0)}")
    print(f"   DD:     {info.get('max_drawdown',0):.1%}")
    print("\n✅ Clean MTF env ready")
