"""
Phase 2 — 1-minute kline backfill for order-flow features
==========================================================
Fetches 3 years of 1m klines per symbol (~1.58M bars each) from the same
Binance USDT-M Futures public REST endpoint data_pipeline.py uses.
Needed by orderflow_features.py to build daily volume profiles (POC, value
area, LVNs, naked POCs).

Run on your Mac (NOT Railway):
    python fetch_1m.py                     # default symbols, 3 years
    python fetch_1m.py --years 2
    python fetch_1m.py --symbols BTCUSDT

Output: ./data/{SYMBOL}_1m.csv  (~150MB each — git-ignored)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from data_pipeline import _get, KLINE_COLS, DATA_DIR

MINUTE_MS = 60_000


def fetch_1m(symbol: str, years: int, force: bool = False) -> None:
    out = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=365 * years)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"[fetch] {symbol} 1m klines ({start:%Y-%m-%d} -> {end:%Y-%m-%d})")
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": "1m",
            "startTime": cursor, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + MINUTE_MS
        if len(rows) % 150_000 < 1500:
            ts = datetime.fromtimestamp(batch[-1][0] / 1000, tz=timezone.utc)
            print(f"    ...{len(rows)} bars so far (at {ts:%Y-%m-%d})")
        time.sleep(0.35)

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    df.to_csv(out, index=False)
    print(f"    saved {len(df)} bars -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    for sym in args.symbols:
        fetch_1m(sym, args.years, args.force)
    print("\nDone. Next: python orderflow_features.py --check")
