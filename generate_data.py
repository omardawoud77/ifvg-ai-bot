"""
generate_data.py
Generates 1000 synthetic IFVG trades and saves them to trades.csv.
Win probability is shaped by: HTF bias alignment, RSI range, volume ratio, session.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

# ── Categorical pools ────────────────────────────────────────────────────────
TIMEFRAMES  = ["15m", "1H", "4H", "Daily"]
SESSIONS    = ["london", "newyork", "asia", "overnight"]
HTF_BIASES  = ["bullish", "bearish"]
DIRECTIONS  = ["long", "short"]

# ── Sample raw features ──────────────────────────────────────────────────────
timeframe       = rng.choice(TIMEFRAMES, N, p=[0.40, 0.30, 0.20, 0.10])
session         = rng.choice(SESSIONS,   N, p=[0.30, 0.35, 0.20, 0.15])
htf_bias        = rng.choice(HTF_BIASES, N)
trade_direction = rng.choice(DIRECTIONS, N)

rsi_at_entry    = rng.uniform(25, 75, N)
ema_diff        = rng.normal(0, 8, N)          # ema9 - ema21 in points
volume_ratio    = rng.lognormal(0.05, 0.35, N) # volume / vol_ma
entry_price     = rng.uniform(4000, 5500, N)   # NQ / SPX range
sl_distance     = rng.uniform(5, 60, N)        # points from entry to SL

# ── Win probability model ────────────────────────────────────────────────────
def win_prob(tf, sess, bias, direction, rsi, vol_ratio, ema_d):
    p = 0.42  # baseline below 50 % — edges must be earned

    # HTF bias aligns with direction
    bias_match = (bias == "bullish" and direction == "long") or \
                 (bias == "bearish" and direction == "short")
    if bias_match:
        p += 0.18

    # RSI in sweet spot (avoid extremes)
    if 45 <= rsi <= 65:
        p += 0.08
    elif rsi < 35 or rsi > 70:
        p -= 0.08

    # Volume confirmation
    if vol_ratio > 1.5:
        p += 0.10
    elif vol_ratio > 1.2:
        p += 0.05
    elif vol_ratio < 0.8:
        p -= 0.07

    # Session quality
    if sess in ("london", "newyork"):
        p += 0.07
    elif sess == "overnight":
        p -= 0.06

    # EMA momentum aligned with direction
    ema_bull = ema_d > 0
    if (direction == "long" and ema_bull) or (direction == "short" and not ema_bull):
        p += 0.06
    else:
        p -= 0.04

    # Higher timeframes carry more signal weight
    if tf == "4H":
        p += 0.04
    elif tf == "Daily":
        p += 0.07
    elif tf == "15m":
        p -= 0.03

    return float(np.clip(p, 0.05, 0.95))

probs = np.array([
    win_prob(timeframe[i], session[i], htf_bias[i], trade_direction[i],
             rsi_at_entry[i], volume_ratio[i], ema_diff[i])
    for i in range(N)
])

rolls   = rng.random(N)
results = np.where(rolls < probs, "win", "loss")

# ── PnL in points (RR ~ 1:1 with noise) ─────────────────────────────────────
rr_noise   = rng.uniform(0.8, 1.4, N)
pnl_points = np.where(
    results == "win",
     sl_distance * rr_noise,
    -sl_distance * rng.uniform(0.6, 1.0, N)   # sometimes stopped early
)
pnl_points = np.round(pnl_points, 2)

# ── Assemble DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame({
    "timeframe":          timeframe,
    "rsi_at_entry":       np.round(rsi_at_entry, 2),
    "ema_diff":           np.round(ema_diff, 2),
    "volume_ratio":       np.round(volume_ratio, 3),
    "session":            session,
    "htf_bias":           htf_bias,
    "trade_direction":    trade_direction,
    "sl_distance_points": np.round(sl_distance, 2),
    "entry_price":        np.round(entry_price, 2),
    "result":             results,
    "pnl_points":         pnl_points,
})

df.to_csv("trades.csv", index=False)

# ── Quick sanity print ───────────────────────────────────────────────────────
wins = (df["result"] == "win").sum()
print(f"Generated {N} trades  —  {wins} wins ({wins/N:.1%} win rate)")
print(df.head(10).to_string(index=False))
print(f"\nSaved → trades.csv")
