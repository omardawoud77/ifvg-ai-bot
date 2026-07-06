"""
Phase 2 — Calibration Analysis (the July-18 session, runnable any day)
======================================================================
Answers the four spec questions from README_PHASE2.md using the shadow
diary + subsequent price data. Read-only; produces a report, changes
nothing.

  1. CALIBRATION — when the gate says X% confident, how often is the
     signal actually right? (overall and per regime/session/tier)
  2. P(SHORT) — did the model want to short, was it blocked, was the
     market shortable?
  3. GATE AUDIT — expectancy of EXECUTEd vs REJECTed signals: is the
     gate adding or destroying edge? Where should thresholds sit?
  4. REGIME — per-regime calibration, informing input-vs-gate decision.

Every signal (executed or blocked) is simulated under the LIVE exit
rules: SL 2.5%, TP 5%, breakeven at 50% TP, trail at 75% TP locking
50% TP, 48h timeout, taker fee both sides.

Usage:
    ./pull_shadow_logs.sh                       # freshest diary first
    python data_pipeline.py --force             # refresh klines to today
    python calibration_analysis.py              # writes report to stdout
    python calibration_analysis.py --min-rows 500   # refuse tiny samples
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "data", "shadow_archive")
DATA = os.path.join(HERE, "data")
SYMBOLS = {"btcusdt": "BTCUSDT", "ethusdt": "ETHUSDT", "solusdt": "SOLUSDT"}

SL_PCT, TP_PCT, MAX_HOLD, FEE = 0.025, 0.05, 48, 0.0005


# ------------------------------------------------------------ simulation
def simulate_signal(bars: pd.DataFrame, start_i: int, direction: int) -> dict | None:
    """Outcome a signal WOULD have had under live exit rules.
    Entry at close of the decision bar; walk forward up to MAX_HOLD bars."""
    if start_i + 1 >= len(bars) or direction == 0:
        return None
    e = float(bars["close"].iloc[start_i]) * (1 + FEE * direction)
    sl = e * (1 - SL_PCT * direction)
    be_set = trail_set = False
    for k in range(1, MAX_HOLD + 1):
        i = start_i + k
        if i >= len(bars):
            return None  # signal too recent — outcome unknown yet
        h, l, c = (float(bars[x].iloc[i]) for x in ("high", "low", "close"))
        hi_move = (h - e) / e if direction == 1 else (e - l) / e
        if (direction == 1 and l <= sl) or (direction == -1 and h >= sl):
            pnl = (sl - e) / e * direction - FEE
            return {"pnl_pct": pnl, "reason": "SL", "bars_held": k}
        if hi_move >= TP_PCT:
            pnl = TP_PCT - FEE
            return {"pnl_pct": pnl, "reason": "TP", "bars_held": k}
        if not be_set and hi_move >= TP_PCT * 0.5:
            be_set, sl = True, e
        if be_set and not trail_set and hi_move >= TP_PCT * 0.75:
            trail_set, sl = True, e * (1 + TP_PCT * 0.5 * direction)
    c = float(bars["close"].iloc[start_i + MAX_HOLD])
    pnl = (c - e) / e * direction - FEE
    return {"pnl_pct": pnl, "reason": "TIMEOUT", "bars_held": MAX_HOLD}


def load_and_simulate() -> pd.DataFrame:
    rows = []
    for sym_l, sym in SYMBOLS.items():
        diary_path = os.path.join(ARCHIVE, f"shadow_log_{sym_l}.csv")
        klines_path = os.path.join(DATA, f"{sym}_1h.csv")
        if not (os.path.exists(diary_path) and os.path.exists(klines_path)):
            print(f"[warn] missing data for {sym}, skipping")
            continue
        diary = pd.read_csv(diary_path)
        # restarts can re-log the same bar — keep the latest decision per bar
        diary = diary.drop_duplicates(subset=["bar_ts"], keep="last")
        bars = pd.read_csv(klines_path)
        bar_ts = pd.to_datetime(bars["open_time"], utc=True, format="ISO8601")
        ts_to_i = {t: i for i, t in enumerate(bar_ts)}

        for _, r in diary.iterrows():
            try:
                t = pd.to_datetime(r["bar_ts"], utc=True, format="ISO8601")
            except (ValueError, TypeError):
                continue
            i = ts_to_i.get(t)
            action = int(r["action"])
            direction = 1 if action == 1 else (-1 if action == 2 else 0)
            if i is None or direction == 0:
                continue
            sim = simulate_signal(bars, i, direction)
            if sim is None:
                continue
            rows.append({
                "symbol": sym, "bar_ts": t, "direction": direction,
                "p_long": float(r["p_long"]), "p_short": float(r["p_short"]),
                "confidence": float(r["confidence"]),
                "verdict": str(r["verdict"]), "tier": str(r["tier"]),
                "regime": str(r["regime"]), "session": str(r["session"]),
                "executed": str(r["verdict"]).startswith(("EXECUTE", "WEAK_EXECUTE")),
                **sim,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ report
def _bucket_table(df: pd.DataFrame, col: str, edges: list[float]) -> str:
    lines = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = df[(df[col] >= lo) & (df[col] < hi)]
        if len(b) == 0:
            continue
        wr = (b["pnl_pct"] > 0).mean()
        lines.append(f"    {col} {lo:.2f}-{hi:.2f}:  n={len(b):>4}  "
                     f"claimed~{(lo + hi) / 2:.0%}  realized_wr={wr:.0%}  "
                     f"exp={b['pnl_pct'].mean():+.3%}")
    return "\n".join(lines) if lines else "    (no rows)"


def _group_table(df: pd.DataFrame, col: str, min_n: int = 10) -> str:
    lines = []
    for val, g in df.groupby(col):
        marker = "" if len(g) >= min_n else "  (small sample!)"
        lines.append(f"    {val:<16} n={len(g):>4}  wr={(g['pnl_pct'] > 0).mean():.0%}  "
                     f"exp={g['pnl_pct'].mean():+.3%}{marker}")
    return "\n".join(lines) if lines else "    (no rows)"


def report(df: pd.DataFrame, min_rows: int):
    n = len(df)
    print(f"CALIBRATION REPORT — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"Simulated signals with known outcomes: {n}\n")
    if n < min_rows:
        print(f"⚠️  INSUFFICIENT DATA: {n} < {min_rows} required rows.")
        print("    Numbers below are illustrative only — do NOT set the spec from them.\n")

    print("── 1. CALIBRATION (gate confidence vs realized) " + "─" * 20)
    print(_bucket_table(df, "confidence", [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]))
    print("\n   directional probability vs outcome:")
    longs = df[df["direction"] == 1]
    shorts = df[df["direction"] == -1]
    if len(longs):
        print(_bucket_table(longs, "p_long", [0.4, 0.5, 0.6, 0.7, 0.8, 1.01]))
    if len(shorts):
        print(_bucket_table(shorts, "p_short", [0.4, 0.5, 0.6, 0.7, 0.8, 1.01]))

    print("\n── 2. P(SHORT) " + "─" * 50)
    print(f"    signals wanting SHORT: {len(shorts)}/{n} ({len(shorts) / max(n, 1):.0%})")
    if len(shorts):
        exec_s = shorts[shorts["executed"]]
        blocked_s = shorts[~shorts["executed"]]
        print(f"    executed: {len(exec_s)} (exp={exec_s['pnl_pct'].mean():+.3%})"
              if len(exec_s) else "    executed: 0")
        print(f"    blocked:  {len(blocked_s)} (exp had they run: "
              f"{blocked_s['pnl_pct'].mean():+.3%})" if len(blocked_s) else "    blocked: 0")

    print("\n── 3. GATE AUDIT (executed vs blocked) " + "─" * 28)
    print(_group_table(df, "executed"))
    print("\n   by verdict:")
    print(_group_table(df, "verdict"))
    print("\n   by tier:")
    print(_group_table(df, "tier"))

    print("\n── 4. REGIME / SESSION " + "─" * 42)
    print(_group_table(df, "regime"))
    print()
    print(_group_table(df, "session"))

    print("\n── EXIT REASONS " + "─" * 48)
    print(df["reason"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=300,
                    help="minimum simulated signals for a spec-grade report")
    args = ap.parse_args()
    df = load_and_simulate()
    if df.empty:
        print("No simulatable signals found — check archive and klines freshness.")
    else:
        report(df, args.min_rows)
