"""
Phase 2 — Order-Flow Features (volume profile / auction structure)
==================================================================
The ATAS-style order-flow concepts, computed from Binance's own data so they
are identical in backtest and live:

  A. Prior-day volume profile (from 1m klines, 40 price bins/day)
     - pos_vs_poc      : signed % distance from close to prior day's POC
     - pos_in_value    : where price sits vs prior day's value area
                         (0 = at VAL, 1 = at VAH; outside [0,1] = out of value)
     - dist_lvn        : % distance to nearest prior-day Low Volume Node —
                         thin shelves price tends to slip through
  B. Unfinished business (naked POCs)
     - dist_naked_above: % distance to nearest untested POC above (magnet)
     - dist_naked_below: % distance to nearest untested POC below
     - naked_count_30d : how many untested POCs remain from last 30 sessions
  C. Aggression / delta (from 1h taker-buy volume — who is hitting market)
     - delta_1h        : taker buy share of volume, scaled to [-1, +1]
     - delta_z_72      : z-score of delta vs trailing 72h (unusual aggression)
     - cvd_slope_24    : normalized cumulative-volume-delta slope over 24h

All causal: profile/naked features use only FULLY COMPLETED prior days;
delta features use only the completed current bar and history.

Usage:
    from orderflow_features import build_orderflow_features
    feats = build_orderflow_features(df_1h, df_1m)   # aligned to df_1h rows

Sanity check on downloaded data:
    python orderflow_features.py --check

NOTE: true large-trade detection (individual big prints) needs aggTrades
tick data — deliberately deferred: the historical download is ~100GB and
the taker-delta features above capture most of the aggression signal.
Revisit only if walk-forward shows the group earns its keep.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

def _to_ts(series: pd.Series) -> pd.Series:
    """open_time may be epoch-ms ints (fetch_1m) or ISO strings (data_pipeline)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True, format="ISO8601")


N_BINS = 40            # price bins per daily profile
VALUE_AREA = 0.70      # standard 70% value area
LVN_FRAC = 0.30        # bin is an LVN if vol < 30% of median bin vol
NAKED_LOOKBACK = 30    # sessions a naked POC stays on the radar
CLIP = 10.0


# ---------------------------------------------------------------- profiles
def _daily_profiles(df_1m: pd.DataFrame) -> pd.DataFrame:
    """One row per completed UTC day: POC, VAH, VAL, LVN price list."""
    m = df_1m.copy()
    m["ts"] = _to_ts(m["open_time"])
    m["date"] = m["ts"].dt.date
    for c in ("high", "low", "close", "volume"):
        m[c] = m[c].astype(float)
    m["px"] = (m["high"] + m["low"]) / 2.0

    out = []
    for date, day in m.groupby("date", sort=True):
        lo, hi = day["low"].min(), day["high"].max()
        if hi <= lo or day["volume"].sum() <= 0:
            continue
        edges = np.linspace(lo, hi, N_BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        vol, _ = np.histogram(day["px"], bins=edges, weights=day["volume"])

        poc_i = int(vol.argmax())
        poc = centers[poc_i]

        # 70% value area: expand from POC toward the heavier neighbor
        total = vol.sum()
        covered = vol[poc_i]
        lo_i = hi_i = poc_i
        while covered < VALUE_AREA * total:
            nxt_lo = vol[lo_i - 1] if lo_i > 0 else -1.0
            nxt_hi = vol[hi_i + 1] if hi_i < N_BINS - 1 else -1.0
            if nxt_hi >= nxt_lo:
                hi_i += 1; covered += max(nxt_hi, 0.0)
            else:
                lo_i -= 1; covered += max(nxt_lo, 0.0)
        val, vah = centers[lo_i], centers[hi_i]

        med = np.median(vol[vol > 0]) if (vol > 0).any() else 0.0
        lvns = centers[(vol < LVN_FRAC * med) & (vol > 0)] if med > 0 else []

        out.append({"date": date, "poc": poc, "vah": vah, "val": val,
                    "day_high": hi, "day_low": lo,
                    "lvns": list(np.round(lvns, 8))})
    return pd.DataFrame(out)


def _naked_pocs_asof(profiles: pd.DataFrame) -> list:
    """For each session i, the list of still-untested POCs from the previous
    NAKED_LOOKBACK sessions, judged with data available BEFORE session i."""
    naked_asof = []
    active = []  # list of (session_idx, poc)
    for i in range(len(profiles)):
        naked_asof.append([p for _, p in active])
        row = profiles.iloc[i]
        # session i tests any active POC inside its high-low range
        active = [(j, p) for j, p in active
                  if not (row["day_low"] <= p <= row["day_high"])]
        active.append((i, row["poc"]))
        active = [(j, p) for j, p in active if i - j < NAKED_LOOKBACK]
    return naked_asof


# ---------------------------------------------------------------- features
def build_orderflow_features(df_1h: pd.DataFrame,
                             df_1m: pd.DataFrame) -> pd.DataFrame:
    h = df_1h.copy()
    h["ts"] = _to_ts(h["open_time"])
    h["date"] = h["ts"].dt.date
    for c in ("close", "volume", "taker_buy_base"):
        h[c] = h[c].astype(float)

    profiles = _daily_profiles(df_1m)
    naked_lists = _naked_pocs_asof(profiles)
    # map: session date -> (profile of PREVIOUS session, naked list as of today)
    prev_profile, naked_map = {}, {}
    for i in range(len(profiles)):
        d = profiles.iloc[i]["date"]
        naked_map[d] = naked_lists[i]
        if i > 0:
            prev_profile[d] = profiles.iloc[i - 1]

    n = len(h)
    cols = {k: np.zeros(n) for k in
            ("pos_vs_poc", "pos_in_value", "dist_lvn",
             "dist_naked_above", "dist_naked_below", "naked_count_30d")}

    closes = h["close"].values
    dates = h["date"].values
    for i in range(n):
        c, d = closes[i], dates[i]
        p = prev_profile.get(d)
        if p is not None and c > 0:
            cols["pos_vs_poc"][i] = (c - p["poc"]) / c
            width = p["vah"] - p["val"]
            cols["pos_in_value"][i] = ((c - p["val"]) / width) if width > 0 else 0.5
            if len(p["lvns"]):
                cols["dist_lvn"][i] = min(abs(c - l) for l in p["lvns"]) / c
            else:
                cols["dist_lvn"][i] = CLIP
        nk = naked_map.get(d, [])
        above = [p_ for p_ in nk if p_ > c]
        below = [p_ for p_ in nk if p_ < c]
        cols["dist_naked_above"][i] = ((min(above) - c) / c) if above else CLIP
        cols["dist_naked_below"][i] = ((c - max(below)) / c) if below else CLIP
        cols["naked_count_30d"][i] = len(nk)

    feats = pd.DataFrame(cols, index=h.index)

    # C. aggression / delta from completed 1h bars
    vol = h["volume"].replace(0, np.nan)
    delta = (2.0 * h["taker_buy_base"] / vol - 1.0)          # [-1, +1]
    feats["delta_1h"] = delta
    mu = delta.rolling(72, min_periods=24).mean()
    sd = delta.rolling(72, min_periods=24).std()
    feats["delta_z_72"] = (delta - mu) / sd
    cvd = (delta * vol).fillna(0).cumsum()
    cvd_move = cvd - cvd.shift(24)
    feats["cvd_slope_24"] = cvd_move / (vol.rolling(24).sum())

    feats["naked_count_30d"] = feats["naked_count_30d"] / NAKED_LOOKBACK
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.clip(lower=-CLIP, upper=CLIP).fillna(0.0)
    return feats


# ---------------------------------------------------------------- check
def _check():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df_1h = pd.read_csv(os.path.join(data_dir, f"{sym}_1h.csv"))
        df_1m = pd.read_csv(os.path.join(data_dir, f"{sym}_1m.csv"))
        feats = build_orderflow_features(df_1h, df_1m)
        print(f"\n{sym}: {len(feats)} rows, {feats.shape[1]} features")
        print(feats.describe().T[["mean", "std", "min", "max"]].round(4))
        assert not feats.isna().any().any(), "NaNs leaked!"
        assert len(feats) == len(df_1h), "row misalignment!"
    print("\nAll checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        print(__doc__)
