"""
Phase 2 — Data Pipeline
=======================
Fetches historical data needed for the v3 retrain from Binance USDT-M Futures
public REST endpoints (no API key required):

  - 1H / 4H / 1D klines for each symbol (default: 3 years)
  - Funding rate history (8h intervals)

Everything is cached to ./data/ as CSV. Re-running is incremental-ish:
if a file exists it is skipped unless --force is passed.

Run on your Mac (NOT Railway):
    python data_pipeline.py                       # default symbols, 3 years
    python data_pipeline.py --years 2
    python data_pipeline.py --symbols BTCUSDT ETHUSDT
    python data_pipeline.py --force               # re-download everything

Public endpoints used:
    GET https://fapi.binance.com/fapi/v1/klines        (limit 1500/req)
    GET https://fapi.binance.com/fapi/v1/fundingRate   (limit 1000/req)

Rate limiting: sleeps 0.35s between requests — far under Binance's public
weight limits, and this runs from your home IP, not the shared Railway one.
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

INTERVALS = {"1h": 3600_000, "4h": 4 * 3600_000, "1d": 24 * 3600_000}

SLEEP = 0.35  # seconds between requests


def _get(path: str, params: dict) -> list:
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            wait = int(r.headers.get("Retry-After", 60))
            print(f"    rate-limited ({r.status_code}), sleeping {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {path} after 5 attempts")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginate through klines from start_ms to end_ms."""
    rows = []
    cursor = start_ms
    step = INTERVALS[interval]
    while cursor < end_ms:
        batch = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor and len(batch) < 1500:
            break
        cursor = last_open + step
        time.sleep(SLEEP)
        if len(rows) % 15000 == 0:
            print(f"    ...{len(rows)} bars so far "
                  f"(at {datetime.fromtimestamp(cursor/1000, tz=timezone.utc):%Y-%m-%d})")
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if df.empty:
        return df
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "quote_volume": float, "trades": int,
        "taker_buy_base": float, "taker_buy_quote": float,
    })
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"]).drop_duplicates(subset="open_time")
    df = df[df["open_time"] < pd.Timestamp.now(tz="UTC").floor(
        {"1h": "h", "4h": "4h", "1d": "D"}[interval])]  # drop unclosed bar
    return df.sort_values("open_time").reset_index(drop=True)


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginate through funding rate history (one row per 8h)."""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if last <= cursor and len(batch) < 1000:
            break
        cursor = last + 1
        time.sleep(SLEEP)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["fundingTime", "fundingRate"]].drop_duplicates(subset="fundingTime")
    return df.sort_values("fundingTime").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    for sym in args.symbols:
        for interval in INTERVALS:
            path = os.path.join(DATA_DIR, f"{sym}_{interval}.csv")
            if os.path.exists(path) and not args.force:
                print(f"[skip] {path} exists")
                continue
            print(f"[fetch] {sym} {interval} klines "
                  f"({start:%Y-%m-%d} → {end:%Y-%m-%d})")
            df = fetch_klines(sym, interval, start_ms, end_ms)
            df.to_csv(path, index=False)
            print(f"    saved {len(df)} bars → {path}")

        fpath = os.path.join(DATA_DIR, f"{sym}_funding.csv")
        if os.path.exists(fpath) and not args.force:
            print(f"[skip] {fpath} exists")
        else:
            print(f"[fetch] {sym} funding rates")
            fdf = fetch_funding(sym, start_ms, end_ms)
            fdf.to_csv(fpath, index=False)
            print(f"    saved {len(fdf)} funding rows → {fpath}")

    print("\nDone. Files in ./data/ — next: python features_v3.py --check")


if __name__ == "__main__":
    main()
