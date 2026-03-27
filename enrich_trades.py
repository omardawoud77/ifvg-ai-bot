"""
enrich_trades.py
─────────────────────────────────────────────────────────────────────────────
Reads trades_export-143.csv (broker export),
downloads NQ=F 1-minute bars from Yahoo Finance for the full date range,
calculates RSI(14), EMA9, EMA21, Volume MA(20) at each trade's entry bar,
and outputs real_trades_enriched.csv ready for train_model.py.

Run:
    pip install pandas numpy yfinance --break-system-packages
    python enrich_trades.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

INPUT_FILE  = "trades_export-143.csv"
OUTPUT_FILE = "real_trades_enriched.csv"
SYMBOL      = "NQ=F"          # Nasdaq futures on Yahoo Finance

# ── Indicator functions ───────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    vol_ma = volume.rolling(period).mean()
    return (volume / vol_ma.replace(0, np.nan)).fillna(1.0)

# ── Session classifier (UTC) ──────────────────────────────────────────────────

def classify_session(dt_utc: pd.Timestamp) -> str:
    minutes = dt_utc.hour * 60 + dt_utc.minute
    # Asia: 00:00 - 08:00 UTC
    if 0 <= minutes < 8 * 60:
        return "asia"
    # London: 08:00 - 13:30 UTC
    if 8 * 60 <= minutes < 13 * 60 + 30:
        return "london"
    # New York: 13:30 - 20:00 UTC
    if 13 * 60 + 30 <= minutes < 20 * 60:
        return "newyork"
    # Overnight: 20:00 - 00:00 UTC
    return "overnight"

# ── Load trades ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  TRADE ENRICHMENT SCRIPT")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    sys.exit(f"Error: {INPUT_FILE} not found. Place it in the same folder.")

print(f"  Loaded {len(df)} trades from {INPUT_FILE}")

# Parse timestamps — format: 07/21/2025 17:59:00 +04:00
df["EnteredAt"] = pd.to_datetime(df["EnteredAt"], utc=True)
df["ExitedAt"]  = pd.to_datetime(df["ExitedAt"],  utc=True)

# Convert to UTC
df["entry_utc"] = df["EnteredAt"].dt.tz_convert("UTC")
df["exit_utc"]  = df["ExitedAt"].dt.tz_convert("UTC")

date_min = df["entry_utc"].min().date()
date_max = df["entry_utc"].max().date()
print(f"  Date range : {date_min}  →  {date_max}")

# ── Download 1-minute bars from Yahoo Finance ─────────────────────────────────
# Yahoo limits 1m data to last 30 days — use 60m for older dates

print(f"\n  Downloading market data for {SYMBOL} ...")

# Split into recent (1m available) and older (use 60m)
cutoff = pd.Timestamp.utcnow() - timedelta(days=29)
cutoff = cutoff.tz_localize(None)

start_str = str(date_min)
end_str   = str(date_max + timedelta(days=1))

# Download 60-minute bars for full range (available for all dates)
print("  Fetching 60m bars (full range) ...")
bars_60m = yf.download(
    SYMBOL,
    start=start_str,
    end=end_str,
    interval="60m",
    auto_adjust=True,
    progress=False,
)

if bars_60m.empty:
    sys.exit("Error: Could not download data from Yahoo Finance. Check internet connection.")

# Flatten multi-level columns if present
if isinstance(bars_60m.columns, pd.MultiIndex):
    bars_60m.columns = bars_60m.columns.get_level_values(0)

bars_60m.index = pd.to_datetime(bars_60m.index, utc=True)
bars_60m = bars_60m.sort_index()

print(f"  Downloaded {len(bars_60m)} hourly bars  ({bars_60m.index[0].date()} → {bars_60m.index[-1].date()})")

# Try to get 1m bars for recent trades (last 29 days)
bars_1m = pd.DataFrame()
recent_trades = df[df["entry_utc"] >= cutoff.tz_localize("UTC")]
if not recent_trades.empty:
    print("  Fetching 1m bars (recent 29 days) ...")
    try:
        bars_1m = yf.download(
            SYMBOL,
            period="29d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(bars_1m.columns, pd.MultiIndex):
            bars_1m.columns = bars_1m.columns.get_level_values(0)
        bars_1m.index = pd.to_datetime(bars_1m.index, utc=True)
        bars_1m = bars_1m.sort_index()
        print(f"  Downloaded {len(bars_1m)} 1-minute bars")
    except Exception as e:
        print(f"  Warning: 1m download failed ({e}), using 60m only")
        bars_1m = pd.DataFrame()

# ── Pre-calculate indicators on 60m bars ─────────────────────────────────────

print("\n  Calculating indicators on 60m bars ...")
bars_60m["ema9"]         = calc_ema(bars_60m["Close"], 9)
bars_60m["ema21"]        = calc_ema(bars_60m["Close"], 21)
bars_60m["ema_diff"]     = bars_60m["ema9"] - bars_60m["ema21"]
bars_60m["rsi"]          = calc_rsi(bars_60m["Close"], 14)
bars_60m["volume_ratio"] = calc_volume_ratio(bars_60m["Volume"], 20)

# HTF bias: 1H candle direction (close vs open)
bars_60m["htf_bias"] = np.where(
    bars_60m["Close"] >= bars_60m["Open"], "bullish", "bearish"
)

# ── Pre-calculate indicators on 1m bars (if available) ───────────────────────

if not bars_1m.empty:
    print("  Calculating indicators on 1m bars ...")
    bars_1m["ema9"]         = calc_ema(bars_1m["Close"], 9)
    bars_1m["ema21"]        = calc_ema(bars_1m["Close"], 21)
    bars_1m["ema_diff"]     = bars_1m["ema9"] - bars_1m["ema21"]
    bars_1m["rsi"]          = calc_rsi(bars_1m["Close"], 14)
    bars_1m["volume_ratio"] = calc_volume_ratio(bars_1m["Volume"], 20)
    bars_1m["htf_bias"]     = np.where(
        bars_1m["Close"] >= bars_1m["Open"], "bullish", "bearish"
    )

# ── Helper: find nearest bar ──────────────────────────────────────────────────

def get_bar_at(bars: pd.DataFrame, ts: pd.Timestamp, tolerance_minutes: int = 5):
    """Return the row of bars closest to ts within tolerance."""
    if bars.empty:
        return None
    # Find nearest index
    idx = bars.index.searchsorted(ts)
    candidates = []
    for offset in [0, -1, 1]:
        i = idx + offset
        if 0 <= i < len(bars):
            diff = abs((bars.index[i] - ts).total_seconds()) / 60
            if diff <= tolerance_minutes:
                candidates.append((diff, i))
    if not candidates:
        return None
    best_i = min(candidates, key=lambda x: x[0])[1]
    return bars.iloc[best_i]

# ── Enrich each trade ─────────────────────────────────────────────────────────

print("\n  Enriching trades ...")

records = []
missing = 0

for _, row in df.iterrows():
    ts = row["entry_utc"]

    # Try 1m first (more accurate), fall back to 60m
    bar = None
    if not bars_1m.empty and ts >= cutoff.tz_localize("UTC"):
        bar = get_bar_at(bars_1m, ts, tolerance_minutes=2)

    if bar is None:
        bar = get_bar_at(bars_60m, ts, tolerance_minutes=65)

    if bar is None:
        missing += 1
        rsi_val    = 50.0
        ema_diff   = 0.0
        vol_ratio  = 1.0
        htf_bias   = "bullish"
    else:
        rsi_val   = round(float(bar["rsi"]),          2)
        ema_diff  = round(float(bar["ema_diff"]),     2)
        vol_ratio = round(float(bar["volume_ratio"]), 3)
        htf_bias  = str(bar["htf_bias"])

    # Trade direction
    direction = row["Type"].strip().lower()

    # PnL points (per contract, direction-aware)
    is_long = direction == "long"
    pnl_pts = round(
        (row["ExitPrice"] - row["EntryPrice"]) if is_long
        else (row["EntryPrice"] - row["ExitPrice"]), 2
    )

    # Result
    result = "win" if row["PnL"] > 0 else "loss"

    # Session
    session = classify_session(ts)

    # SL distance (abs price diff as proxy)
    sl_dist = round(abs(row["ExitPrice"] - row["EntryPrice"]), 2)

    # Duration
    duration_td = row["ExitedAt"] - row["EnteredAt"]
    duration_s  = duration_td.total_seconds()

    records.append({
        "timeframe":          "1H",           # your primary timeframe
        "rsi_at_entry":       rsi_val,
        "ema_diff":           ema_diff,
        "volume_ratio":       vol_ratio,
        "session":            session,
        "htf_bias":           htf_bias,
        "trade_direction":    direction,
        "sl_distance_points": sl_dist,
        "entry_price":        round(row["EntryPrice"], 2),
        "result":             result,
        "pnl_points":         pnl_pts,
        # context
        "EnteredAt":          row["EnteredAt"],
        "ExitedAt":           row["ExitedAt"],
        "ExitPrice":          round(row["ExitPrice"], 2),
        "PnL":                row["PnL"],
        "Size":               row["Size"],
        "duration_seconds":   round(duration_s, 1),
        "contract":           row["ContractName"],
    })

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────

total    = len(df_out)
wins     = (df_out["result"] == "win").sum()
win_rate = wins / total if total else 0
avg_pnl  = df_out["PnL"].mean()

session_counts   = df_out["session"].value_counts().to_dict()
direction_counts = df_out["trade_direction"].value_counts().to_dict()
bias_counts      = df_out["htf_bias"].value_counts().to_dict()

print("\n" + "=" * 60)
print("  ENRICHMENT SUMMARY")
print("=" * 60)
print(f"  Total trades    : {total}")
print(f"  Bars not found  : {missing}  (placeholders used)")
print(f"  Wins / Losses   : {wins} / {total - wins}")
print(f"  Win rate        : {win_rate:.1%}")
print(f"  Avg PnL         : {avg_pnl:+.2f}")
print("-" * 60)
print("  Sessions:")
for sess in ("london", "newyork", "asia", "overnight"):
    n = session_counts.get(sess, 0)
    pct = n / total if total else 0
    print(f"    {sess:<12} {n:>4}  ({pct:.0%})")
print("  Directions:")
for d, n in direction_counts.items():
    print(f"    {d:<12} {n:>4}  ({n/total:.0%})")
print("  HTF Bias:")
for b, n in bias_counts.items():
    print(f"    {b:<12} {n:>4}  ({n/total:.0%})")
print("=" * 60)
print(f"\n  Saved → {OUTPUT_FILE}")
print("  Next step: copy this file to ai-model/ and run train_model.py\n")
