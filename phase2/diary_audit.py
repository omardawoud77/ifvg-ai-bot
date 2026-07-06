"""
Phase 2 — Shadow Diary Audit (the July-11 spot-check, runnable any day)
=======================================================================
Health check for shadow-log collection. Safe: read-only, never touches
the live bot.

Checks:
  1. Row counts per symbol vs expected (24/day since collection start)
  2. Hour gaps (bot downtime / logging failures)
  3. Data quality: probabilities populated & sum ~1, verdict/tier/conditions
     non-empty, p_short distribution (the July-18 headline question)
  4. Freshness: newest row age

Usage:
    python diary_audit.py            # audits the local archive
    (run pull_shadow_logs.sh first for the freshest data)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "data", "shadow_archive")
SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
COLLECTION_START = datetime(2026, 7, 4, 22, 46, tzinfo=timezone.utc)


def audit_symbol(sym: str) -> dict:
    path = os.path.join(ARCHIVE, f"shadow_log_{sym}.csv")
    if not os.path.exists(path):
        return {"symbol": sym, "status": "MISSING", "rows": 0}

    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)

    now = datetime.now(timezone.utc)
    # restarts can log the same bar twice — coverage counts DISTINCT bars
    # over the window since the first archived row
    distinct_bars = df["bar_ts"].nunique()
    first_ts = min(df["ts"].iloc[0], COLLECTION_START)
    hours_elapsed = max((now - first_ts).total_seconds() / 3600, 1)
    expected = int(hours_elapsed)
    coverage = distinct_bars / expected if expected else 0

    # hour gaps > 90 min between consecutive rows
    deltas = df["ts"].diff().dt.total_seconds().div(3600).fillna(1)
    gaps = df.loc[deltas > 1.5, "ts"]
    gap_list = [(t.strftime("%m-%d %H:%M"), round(d, 1))
                for t, d in zip(gaps, deltas[deltas > 1.5])]

    # data quality
    probs = df[["p_hold", "p_long", "p_short", "p_close"]].astype(float)
    prob_ok = ((probs.sum(axis=1) - 1).abs() < 0.05).mean()
    quality = {
        "prob_rows_sum_to_1": f"{prob_ok:.0%}",
        "verdict_filled": f"{(df['verdict'].astype(str).str.len() > 0).mean():.0%}",
        "tier_filled": f"{(df['tier'].astype(str).str.len() > 0).mean():.0%}",
        "regime_filled": f"{(df['regime'].astype(str).str.len() > 0).mean():.0%}",
    }
    p_short = probs["p_short"]
    freshness_h = (now - df["ts"].iloc[-1]).total_seconds() / 3600

    return {
        "symbol": sym, "status": "OK", "rows": len(df),
        "expected": expected, "coverage": coverage,
        "gaps": gap_list, "quality": quality,
        "p_short_mean": float(p_short.mean()),
        "p_short_over_40pct": int((p_short > 0.40).sum()),
        "verdict_counts": df["verdict"].value_counts().to_dict(),
        "freshness_hours": freshness_h,
    }


def main():
    print(f"Shadow diary audit — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"Collection started {COLLECTION_START:%Y-%m-%d %H:%M} UTC\n")
    problems = []
    for sym in SYMBOLS:
        r = audit_symbol(sym)
        if r["status"] == "MISSING":
            print(f"{sym.upper()}: ❌ ARCHIVE FILE MISSING")
            problems.append(f"{sym}: archive missing")
            continue
        flag = "✅" if r["coverage"] >= 0.8 else "⚠️"
        print(f"{sym.upper()}: {flag} {r['rows']} rows "
              f"(distinct bars vs ~{r['expected']} expected: "
              f"coverage {r['coverage']:.0%}) "
              f"| newest {r['freshness_hours']:.1f}h old")
        if r["coverage"] < 0.8:
            problems.append(f"{sym}: coverage {r['coverage']:.0%} < 80%")
        if r["freshness_hours"] > 3:
            problems.append(f"{sym}: newest row {r['freshness_hours']:.1f}h old")
        if r["gaps"]:
            print(f"  gaps: {r['gaps'][:5]}{' ...' if len(r['gaps']) > 5 else ''}")
        print(f"  quality: {r['quality']}")
        print(f"  p_short: mean={r['p_short_mean']:.3f}, "
              f"rows with p_short>0.40: {r['p_short_over_40pct']}")
        print(f"  verdicts: {r['verdict_counts']}\n")

    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  ⚠️ {p}")
    else:
        print("ALL CHECKS PASSED — collection on track for the calibration.")


if __name__ == "__main__":
    main()
