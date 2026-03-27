"""
convert_real_trades.py
Converts a raw broker export (trades_export.csv) into the format expected
by train_model.py / predict.py and saves it as real_trades.csv.
"""

import sys
import pandas as pd

INPUT_FILE  = "trades_export.csv"
OUTPUT_FILE = "real_trades.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    sys.exit(f"Error: {INPUT_FILE} not found. Place it in the ai-model folder.")

required_cols = {"EnteredAt", "ExitedAt", "EntryPrice", "ExitPrice",
                 "PnL", "Size", "Type", "TradeDuration"}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"Error: missing columns in {INPUT_FILE}: {missing}")

# ── Parse timestamps ──────────────────────────────────────────────────────────
df["EnteredAt"] = pd.to_datetime(df["EnteredAt"], utc=True)
df["ExitedAt"]  = pd.to_datetime(df["ExitedAt"],  utc=True)

# Shift to UTC+2 for session classification
df["entry_local"] = df["EnteredAt"].dt.tz_convert("Etc/GMT-2")

# ── Derived: trade_direction ──────────────────────────────────────────────────
df["trade_direction"] = df["Type"].str.strip().str.lower()

# ── Derived: pnl_points ───────────────────────────────────────────────────────
is_long = df["trade_direction"] == "long"
df["pnl_points"] = (
    (df["ExitPrice"] - df["EntryPrice"]).where(is_long,
     df["EntryPrice"] - df["ExitPrice"])
).round(2)

# ── Derived: result ───────────────────────────────────────────────────────────
df["result"] = df["PnL"].apply(lambda x: "win" if x > 0 else "loss")

# ── Derived: session (UTC+2 local hour) ──────────────────────────────────────
def classify_session(hour: int) -> str:
    if 10 <= hour < 13:
        return "london"
    if 15 <= hour < 21 or (hour == 21 and 0 == 0):  # 15:30 approximated to hour
        return "newyork"
    if 3 <= hour < 9:
        return "asia"
    return "overnight"

# Refine newyork to start at 15:30
def classify_session_precise(dt) -> str:
    h, m = dt.hour, dt.minute
    minutes = h * 60 + m
    if 10 * 60 <= minutes < 13 * 60:
        return "london"
    if 15 * 60 + 30 <= minutes < 21 * 60:
        return "newyork"
    if 3 * 60 <= minutes < 9 * 60:
        return "asia"
    return "overnight"

df["session"] = df["entry_local"].apply(classify_session_precise)

# ── Derived: duration_seconds ─────────────────────────────────────────────────
def parse_duration(val) -> float:
    """
    Accepts multiple formats:
      - numeric seconds (int/float)
      - 'HH:MM:SS' or 'MM:SS'
      - pandas Timedelta string e.g. '0 days 00:05:30'
    Returns total seconds as float.
    """
    if pd.isna(val):
        return float("nan")
    # Already numeric
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    s = str(val).strip()
    # pandas Timedelta string
    try:
        return pd.Timedelta(s).total_seconds()
    except Exception:
        pass
    # HH:MM:SS or MM:SS
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except ValueError:
        pass
    return float("nan")

df["duration_seconds"] = df["TradeDuration"].apply(parse_duration)

# ── Placeholder columns ───────────────────────────────────────────────────────
df["timeframe"]          = "1H"
df["rsi_at_entry"]       = 50.0
df["ema_diff"]           = 0.0
df["volume_ratio"]       = 1.0
df["htf_bias"]           = "bullish"
df["sl_distance_points"] = 20.0
df["entry_price"]        = df["EntryPrice"]

# ── Select & order output columns ─────────────────────────────────────────────
out_cols = [
    "timeframe", "rsi_at_entry", "ema_diff", "volume_ratio",
    "session", "htf_bias", "trade_direction",
    "sl_distance_points", "entry_price",
    "result", "pnl_points",
    # bonus context columns kept for reference
    "EnteredAt", "ExitedAt", "ExitPrice", "PnL", "Size", "duration_seconds",
]
df_out = df[out_cols]

df_out.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
total      = len(df_out)
wins       = (df_out["result"] == "win").sum()
win_rate   = wins / total if total else 0
avg_pnl    = df_out["PnL"].mean()
best       = df_out["PnL"].max()
worst      = df_out["PnL"].min()
best_pts   = df_out["pnl_points"].max()
worst_pts  = df_out["pnl_points"].min()

session_counts = df_out["session"].value_counts().to_dict()
direction_counts = df_out["trade_direction"].value_counts().to_dict()

print("\n" + "=" * 52)
print("  REAL TRADES CONVERSION SUMMARY")
print("=" * 52)
print(f"  Total trades   : {total}")
print(f"  Wins / Losses  : {wins} / {total - wins}")
print(f"  Win rate       : {win_rate:.1%}")
print(f"  Avg PnL        : {avg_pnl:+.2f}")
print(f"  Best trade     : {best:+.2f}  ({best_pts:+.2f} pts)")
print(f"  Worst trade    : {worst:+.2f}  ({worst_pts:+.2f} pts)")
print("-" * 52)
print("  Sessions:")
for sess in ("london", "newyork", "asia", "overnight"):
    n = session_counts.get(sess, 0)
    print(f"    {sess:<12} {n:>4}  ({n/total:.0%})")
print("  Directions:")
for d, n in direction_counts.items():
    print(f"    {d:<12} {n:>4}  ({n/total:.0%})")
print("=" * 52)
print(f"\nSaved → {OUTPUT_FILE}\n")
