"""
Phase 2 — Feature Engineering v3
================================
The new observation features for the retrain, computed on 1H OHLCV data.
These are ADDITIVE to the existing 44-feature set — the v3 environment
concatenates [existing_44, features_v3] into the observation.

Feature groups (all causal — nothing peeks at future bars):

  A. Market structure (swing pivots)
     - dist_swing_high / dist_swing_low : % distance to nearest confirmed
       swing high/low (support/resistance proximity)
     - structure_state : +1 higher-highs-higher-lows, -1 lower-lows-lower-
       highs, 0 mixed — the "is it making HH/HL or LH/LL" question
  B. Prior-day levels
     - pos_in_prev_range : where price sits in yesterday's high-low range
       (0 = at prev low, 1 = at prev high, can exceed [0,1] on breakouts)
     - dist_prev_high / dist_prev_low : signed % distance to prior day H/L
  C. Funding / basis
     - funding_now : current 8h funding rate (as-of joined, no lookahead)
     - funding_z : z-score of funding vs trailing 30d — extremes precede
       squeezes
     - funding_cum_3d : cumulative funding over 3 days (persistent crowding)
  D. BTC-lead (ETH/SOL only; zeros for BTC itself)
     - btc_ret_1 / btc_ret_3 / btc_ret_12 : BTC returns over 1/3/12 bars
     - btc_corr_72 : 72-bar rolling correlation of returns with BTC
     - rel_strength_24 : symbol 24-bar return minus BTC 24-bar return

Usage:
    from features_v3 import build_features_v3
    feats = build_features_v3(df_1h, funding_df, btc_df_1h)  # btc_df=None for BTC

Sanity check on downloaded data:
    python features_v3.py --check
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PIVOT_WIDTH = 5          # bars each side to confirm a swing pivot
FUNDING_Z_WINDOW = 90    # 90 funding periods = 30 days of 8h funding
V3_FEATURE_NAMES = [
    "dist_swing_high", "dist_swing_low", "structure_state",
    "pos_in_prev_range", "dist_prev_high", "dist_prev_low",
    "funding_now", "funding_z", "funding_cum_3d",
    "btc_ret_1", "btc_ret_3", "btc_ret_12", "btc_corr_72", "rel_strength_24",
]


# ---------------------------------------------------------------- structure
def _swing_pivots(df: pd.DataFrame, width: int = PIVOT_WIDTH):
    """Confirmed swing highs/lows. A pivot at bar i is only KNOWN at bar
    i+width (needs `width` bars after it) — we shift accordingly so the
    feature is causal."""
    high, low = df["high"].values, df["low"].values
    n = len(df)
    is_high = np.zeros(n, bool)
    is_low = np.zeros(n, bool)
    for i in range(width, n - width):
        if high[i] == high[i - width:i + width + 1].max():
            is_high[i] = True
        if low[i] == low[i - width:i + width + 1].min():
            is_low[i] = True
    return is_high, is_low


def market_structure(df: pd.DataFrame, width: int = PIVOT_WIDTH) -> pd.DataFrame:
    is_high, is_low = _swing_pivots(df, width)
    n = len(df)
    close = df["close"].values
    high, low = df["high"].values, df["low"].values

    last_sh = np.full(n, np.nan)   # most recent confirmed swing high price
    last_sl = np.full(n, np.nan)
    prev_sh = np.full(n, np.nan)   # the one before it (for HH/LH judgement)
    prev_sl = np.full(n, np.nan)

    cur_sh = cur_sh_prev = cur_sl = cur_sl_prev = np.nan
    for i in range(n):
        # pivot at bar i-width becomes KNOWN at bar i
        j = i - width
        if j >= 0:
            if is_high[j]:
                cur_sh_prev, cur_sh = cur_sh, high[j]
            if is_low[j]:
                cur_sl_prev, cur_sl = cur_sl, low[j]
        last_sh[i], prev_sh[i] = cur_sh, cur_sh_prev
        last_sl[i], prev_sl[i] = cur_sl, cur_sl_prev

    out = pd.DataFrame(index=df.index)
    out["dist_swing_high"] = (last_sh - close) / close   # + = resistance above
    out["dist_swing_low"] = (close - last_sl) / close    # + = support below

    hh = last_sh > prev_sh
    hl = last_sl > prev_sl
    lh = last_sh < prev_sh
    ll = last_sl < prev_sl
    state = np.zeros(n)
    state[hh & hl] = 1.0
    state[lh & ll] = -1.0
    out["structure_state"] = state
    return out


# ---------------------------------------------------------------- prev day
def prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Prior COMPLETED UTC day's high/low, forward-filled through today."""
    ts = pd.to_datetime(df["open_time"], utc=True)
    day = ts.dt.floor("D")
    daily_high = df.groupby(day)["high"].max()
    daily_low = df.groupby(day)["low"].min()
    prev_high = day.map(daily_high.shift(1))
    prev_low = day.map(daily_low.shift(1))

    close = df["close"]
    rng = (prev_high - prev_low).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["pos_in_prev_range"] = (close - prev_low) / rng
    out["dist_prev_high"] = (prev_high - close) / close
    out["dist_prev_low"] = (close - prev_low) / close
    return out


# ---------------------------------------------------------------- funding
def funding_features(df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """As-of join funding onto hourly bars (last funding rate known at each
    bar's open) + z-score and 3-day cumulative."""
    out = pd.DataFrame(index=df.index)
    if funding is None or funding.empty:
        out["funding_now"] = 0.0
        out["funding_z"] = 0.0
        out["funding_cum_3d"] = 0.0
        return out

    f = funding.sort_values("fundingTime").copy()
    f["funding_z"] = (
        (f["fundingRate"] - f["fundingRate"].rolling(FUNDING_Z_WINDOW).mean())
        / f["fundingRate"].rolling(FUNDING_Z_WINDOW).std()
    )
    f["funding_cum_3d"] = f["fundingRate"].rolling(9).sum()  # 9 × 8h = 3d

    bars = pd.DataFrame({"open_time": pd.to_datetime(df["open_time"], utc=True)})
    merged = pd.merge_asof(
        bars.sort_values("open_time"), f,
        left_on="open_time", right_on="fundingTime", direction="backward",
    )
    out["funding_now"] = merged["fundingRate"].values
    out["funding_z"] = merged["funding_z"].values
    out["funding_cum_3d"] = merged["funding_cum_3d"].values
    return out


# ---------------------------------------------------------------- BTC lead
def btc_lead_features(df: pd.DataFrame, btc: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cols = ["btc_ret_1", "btc_ret_3", "btc_ret_12", "btc_corr_72",
            "rel_strength_24"]
    if btc is None:                      # the symbol IS BTC
        for c in cols:
            out[c] = 0.0
        return out

    b = btc[["open_time", "close"]].rename(columns={"close": "btc_close"})
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    bars = df[["open_time", "close"]].copy()
    bars["open_time"] = pd.to_datetime(bars["open_time"], utc=True)
    m = pd.merge_asof(bars.sort_values("open_time"), b.sort_values("open_time"),
                      on="open_time", direction="backward")

    bc = m["btc_close"]
    out["btc_ret_1"] = bc.pct_change(1).values
    out["btc_ret_3"] = bc.pct_change(3).values
    out["btc_ret_12"] = bc.pct_change(12).values

    sym_ret = m["close"].pct_change()
    btc_ret = bc.pct_change()
    out["btc_corr_72"] = sym_ret.rolling(72).corr(btc_ret).values
    out["rel_strength_24"] = (m["close"].pct_change(24) - bc.pct_change(24)).values
    return out


# ---------------------------------------------------------------- assemble
def build_features_v3(df_1h: pd.DataFrame,
                      funding: pd.DataFrame | None,
                      btc_1h: pd.DataFrame | None) -> pd.DataFrame:
    """Returns a DataFrame with the 14 v3 features, aligned to df_1h rows.
    Pass btc_1h=None when df_1h IS BTC."""
    parts = [
        market_structure(df_1h),
        prev_day_levels(df_1h),
        funding_features(df_1h, funding),
        btc_lead_features(df_1h, btc_1h),
    ]
    feats = pd.concat(parts, axis=1)[V3_FEATURE_NAMES]
    # Clip extremes and fill warm-up NaNs so the env never sees inf/NaN.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.clip(lower=-10, upper=10).fillna(0.0)
    return feats


# ---------------------------------------------------------------- check
def _check():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    btc = pd.read_csv(os.path.join(data_dir, "BTCUSDT_1h.csv"))
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = pd.read_csv(os.path.join(data_dir, f"{sym}_1h.csv"))
        fpath = os.path.join(data_dir, f"{sym}_funding.csv")
        funding = pd.read_csv(fpath) if os.path.exists(fpath) else None
        if funding is not None:
            funding["fundingTime"] = pd.to_datetime(
                funding["fundingTime"], utc=True, format="ISO8601"
            )
        feats = build_features_v3(df, funding, None if sym == "BTCUSDT" else btc)
        print(f"\n{sym}: {len(feats)} rows, {feats.shape[1]} features")
        print(feats.describe().T[["mean", "std", "min", "max"]].round(4))
        assert not feats.isna().any().any(), "NaNs leaked!"
    print("\nAll checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        print(__doc__)
