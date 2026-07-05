"""
Phase 2 — Crypto RL Environment v3
==================================
Drop-in evolution of crypto/crypto_env_v2.CryptoTechEnv (44 features):

  obs = [existing_44 | features_v3_14 | orderflow_9]  ->  67-dim

What changes vs v2 (each locked in by the Phase 2 README):

  1. LIVE EXIT MECHANICS replicated exactly (v2 trained with only a 2% SL —
     the live bot actually runs tiered SL / TP / breakeven / trail):
       - SL: tiered 2.0-3.0% (default 2.5% = reasoning_engine default)
       - TP: 5% (reasoning_engine get_dynamic_sl_tp default)
       - Breakeven: SL -> entry when price reaches 50% of TP distance
       - Trail: SL -> entry+50%TP when price reaches 75% of TP distance
       - Max hold: 48 bars. Taker fee both sides.
  2. CAUSAL multi-timeframe view. v2's trainer forward-filled COMPLETED
     4h/1d/1w bars by open time — each 1h bar saw the finished HTF candle
     hours before it closed (lookahead). v3 reconstructs the PARTIAL HTF
     candle exactly as it existed at that hour's close — what live sees.
  3. BALANCED REGIME SAMPLING. Episodes are drawn proportionally from
     bull/bear/sideways segments (200-EMA slope on 1D), so the agent can't
     just learn "always long the bull market".

Reward is inherited from v2 (continuous uPnL delta + realized on close).
Final reward shaping (SHORT asymmetry etc.) is a July-18 calibration
decision — subclass and override step() the way CryptoRREnv does.

Usage:
    from crypto_env_v3 import build_mtf_dataset, build_feature_matrix, CryptoEnvV3
    df    = build_mtf_dataset("BTCUSDT")
    feats = build_feature_matrix("BTCUSDT", df)
    env   = CryptoEnvV3(df, feats, episode_len=720)

Smoke test:  python crypto_env_v3.py --check
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "..", "crypto"))

from crypto_env_v2 import CryptoTechEnv                    # noqa: E402
from features_v3 import build_features_v3                  # noqa: E402
from orderflow_features import build_orderflow_features, _to_ts  # noqa: E402

N_V3 = 14
N_OF = 9
N_TOTAL = 44 + N_V3 + N_OF


# ---------------------------------------------------------------- dataset
def _partial_htf(df: pd.DataFrame, prefix: str, period_starts: pd.Series) -> pd.DataFrame:
    """HTF candle AS IT EXISTED at each 1h close: open of the period's first
    bar, cumulative high/low/volume so far, close = current 1h close."""
    g = df.groupby(period_starts.values)
    return pd.DataFrame({
        f"{prefix}_open":   g["open"].transform("first"),
        f"{prefix}_high":   g["high"].cummax(),
        f"{prefix}_low":    g["low"].cummin(),
        f"{prefix}_close":  df["close"],
        f"{prefix}_volume": g["volume"].cumsum(),
    })


def build_mtf_dataset(symbol: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """1h OHLCV + causal partial 4h/1d/1w columns, same layout v2 expects."""
    df = pd.read_csv(os.path.join(data_dir, f"{symbol}_1h.csv"))
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    ts = _to_ts(df["open_time"])
    df["Datetime"] = ts

    h4_start = ts.dt.floor("4h")
    d1_start = ts.dt.floor("D")
    w1_start = d1_start - pd.to_timedelta(ts.dt.dayofweek, unit="D")

    out = pd.concat([
        df[["Datetime", "open", "high", "low", "close", "volume",
            "taker_buy_base", "open_time"]],
        _partial_htf(df, "h4", h4_start),
        _partial_htf(df, "d1", d1_start),
        _partial_htf(df, "w1", w1_start),
    ], axis=1)
    return out.reset_index(drop=True)


def build_feature_matrix(symbol: str, df_mtf: pd.DataFrame,
                         data_dir: str = DATA_DIR) -> np.ndarray:
    """[features_v3 | orderflow] rows aligned 1:1 with df_mtf."""
    df_1h = pd.read_csv(os.path.join(data_dir, f"{symbol}_1h.csv"))
    df_1m = pd.read_csv(os.path.join(data_dir, f"{symbol}_1m.csv"))

    fpath = os.path.join(data_dir, f"{symbol}_funding.csv")
    funding = pd.read_csv(fpath) if os.path.exists(fpath) else None
    if funding is not None:
        funding["fundingTime"] = pd.to_datetime(
            funding["fundingTime"], utc=True, format="ISO8601")

    btc = None
    if symbol != "BTCUSDT":
        btc = pd.read_csv(os.path.join(data_dir, "BTCUSDT_1h.csv"))

    f_v3 = build_features_v3(df_1h, funding, btc)
    f_of = build_orderflow_features(df_1h, df_1m)
    assert len(f_v3) == len(f_of) == len(df_mtf), "feature/dataset row mismatch"

    mat = np.concatenate([f_v3.values, f_of.values], axis=1).astype(np.float32)
    return np.clip(mat, -10.0, 10.0)


def label_regimes(df_mtf: pd.DataFrame) -> np.ndarray:
    """Per-row regime label from 200-EMA slope on the DAILY closes:
    +1 bull, -1 bear, 0 sideways. Used for episode sampling only (not obs)."""
    daily = (df_mtf.set_index("Datetime")["close"]
             .resample("1D").last().dropna())
    ema = daily.ewm(span=200, min_periods=30).mean()
    slope = (ema.pct_change(5) / 5).values   # avg daily EMA slope over a week
    lab = np.where(slope > 0.0005, 1, np.where(slope < -0.0005, -1, 0))
    day_map = dict(zip(daily.index.date, lab))
    days = df_mtf["Datetime"].dt.date
    return np.array([day_map.get(d, 0) for d in days], dtype=int)


# ---------------------------------------------------------------- env
class CryptoEnvV3(CryptoTechEnv):
    """67-feature env with live exit mechanics + regime-balanced episodes."""

    def __init__(self, df: pd.DataFrame, feature_matrix: np.ndarray,
                 episode_len: int = 720,
                 sl_pct: float = 0.025, tp_pct: float = 0.05,
                 fee_pct: float = 0.0005,
                 seed: int | None = None, **kwargs):
        kwargs.setdefault("max_hold_bars", 48)
        super().__init__(df, sl_pct=sl_pct, fee_pct=fee_pct, **kwargs)
        assert feature_matrix.shape == (len(df), N_V3 + N_OF), \
            f"feature matrix {feature_matrix.shape} != ({len(df)}, {N_V3 + N_OF})"
        self.feature_matrix = feature_matrix
        self.tp_pct = tp_pct
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

        self.N_FEATURES = N_TOTAL
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N_TOTAL,), dtype=np.float32)

        self.regimes = label_regimes(df)
        warm = self.start_idx
        last_ok = len(df) - episode_len - 2
        self.regime_pools = {
            r: [i for i in range(warm, max(warm + 1, last_ok))
                if self.regimes[i] == r]
            for r in (-1, 0, 1)
        }
        self._episode_end = len(df) - 1
        self._be_set = False
        self._trail_set = False
        self._sl_price = 0.0

    # -- observation: parent's 44 + precomputed 23 ------------------------
    def _get_obs(self, i):
        saved = self.N_FEATURES
        self.N_FEATURES = 44
        base = super()._get_obs(i)
        self.N_FEATURES = saved
        obs = np.concatenate([base, self.feature_matrix[i]])
        assert len(obs) == N_TOTAL
        return obs.astype(np.float32)

    # -- regime-balanced episode reset ------------------------------------
    def reset(self, seed=None, options=None):
        options = options or {}
        self._reset_state()
        self._be_set = self._trail_set = False
        self._sl_price = 0.0

        if "start_idx" in options:                      # deterministic (walk-forward eval)
            start = int(options["start_idx"])
        else:                                           # balanced regime sampling
            pools = [p for p in self.regime_pools.values() if p]
            pool = pools[int(self.rng.integers(len(pools)))]
            start = int(pool[int(self.rng.integers(len(pool)))])
        self.current_idx = start
        self._episode_end = min(start + self.episode_len, len(self.df) - 1)
        return self._get_obs(self.current_idx), {}

    # -- live exit mechanics ----------------------------------------------
    def _set_entry(self, direction: int, c: float):
        self.position = direction
        self.entry_price = c * (1 + self.fee_pct * direction)
        self.bars_held = 0
        self.total_trades += 1
        self._be_set = self._trail_set = False
        self._sl_price = self.entry_price * (1 - self.sl_pct * direction)

    def step(self, action):
        df = self.df
        i = self.current_idx
        c = float(df["close"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])

        reward = -self.hold_cost
        info = {}
        done = False

        if self.position == 0:
            if action == 1:
                self._set_entry(1, c)
            elif action == 2:
                self._set_entry(-1, c)

        elif self.position != 0:
            self.bars_held += 1
            e = self.entry_price
            d = self.position

            upnl_pct = (c - e) / e * d
            prev_upnl = self.unrealized_pnl
            self.unrealized_pnl = upnl_pct * self.capital * self.trade_pct
            reward += (self.unrealized_pnl - prev_upnl) / self.capital

            close_reason, c_exit = None, 0.0
            hi_move = (h - e) / e * d if d == 1 else (e - l) / e
            # pessimistic intrabar ordering: stop checked before target
            if (d == 1 and l <= self._sl_price) or (d == -1 and h >= self._sl_price):
                close_reason, c_exit = "SL", self._sl_price
            elif hi_move >= self.tp_pct:
                close_reason, c_exit = "TP", e * (1 + self.tp_pct * d)
            else:
                # breakeven / trail arming for FUTURE bars (live re-checks each cycle)
                if not self._be_set and hi_move >= self.tp_pct * 0.5:
                    self._be_set = True
                    self._sl_price = e
                if self._be_set and not self._trail_set and hi_move >= self.tp_pct * 0.75:
                    self._trail_set = True
                    self._sl_price = e * (1 + self.tp_pct * 0.5 * d)

            if action == 3 and close_reason is None:
                close_reason = "AGENT"
                c_exit = c * (1 - self.fee_pct * d)
            if self.bars_held >= self.max_hold_bars and close_reason is None:
                close_reason, c_exit = "TIMEOUT", c
                reward -= 0.01

            if close_reason:
                pnl_pct = (c_exit - e) / e * d - self.fee_pct
                pnl_dollar = pnl_pct * self.capital * self.trade_pct
                self.capital += pnl_dollar
                reward += pnl_pct * 5
                self.wins += pnl_pct > 0
                self.losses += pnl_pct <= 0
                self.peak_capital = max(self.peak_capital, self.capital)
                dd = (self.peak_capital - self.capital) / (self.peak_capital + 1e-8)
                self.max_drawdown = max(self.max_drawdown, dd)
                info.update(trade_closed=True, pnl_pct=pnl_pct,
                            close_reason=close_reason)
                self.position = 0
                self.entry_price = self.unrealized_pnl = 0.0
                self.bars_held = 0
                self._be_set = self._trail_set = False
                self._sl_price = 0.0

        self.current_idx += 1
        if self.current_idx >= self._episode_end:
            done = True
            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct_total = (self.capital - self.initial_capital) / self.initial_capital
            reward += pnl_pct_total * 10 + wr * 2
            info.update(total_pnl=self.capital - self.initial_capital,
                        total_pnl_pct=pnl_pct_total, win_rate=wr,
                        wins=self.wins, losses=self.losses,
                        max_drawdown=self.max_drawdown,
                        total_trades=self.total_trades)

        return (self._get_obs(min(self.current_idx, len(df) - 1)),
                reward, done, False, info)


# ---------------------------------------------------------------- check
def _check():
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = build_mtf_dataset(sym)
        feats = build_feature_matrix(sym, df)
        env = CryptoEnvV3(df, feats, episode_len=720, seed=42)
        pools = {r: len(p) for r, p in env.regime_pools.items()}
        print(f"\n{sym}: {len(df)} bars | regime pools "
              f"bull={pools[1]} sideways={pools[0]} bear={pools[-1]}")

        for ep in range(3):
            obs, _ = env.reset()
            assert obs.shape == (N_TOTAL,) and np.isfinite(obs).all()
            done, steps, closed = False, 0, 0
            while not done:
                obs, r, done, _, inf = env.step(env.action_space.sample())
                assert np.isfinite(obs).all() and np.isfinite(r)
                steps += 1
                closed += inf.get("trade_closed", False)
            print(f"  ep{ep + 1}: {steps} steps | trades={inf.get('total_trades')} "
                  f"| wr={inf.get('win_rate', 0):.0%} "
                  f"| pnl={inf.get('total_pnl_pct', 0):+.1%} "
                  f"| dd={inf.get('max_drawdown', 0):.1%}")
    print("\nAll env v3 checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        print(__doc__)
