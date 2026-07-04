"""
Weekly data updater — run every week to accumulate 5m NQ history.
Each run downloads latest 60d and merges with existing archive.
Run: python3 update_dataset.py
"""
import yfinance as yf, pandas as pd, os
from datetime import datetime

ARCHIVE = 'nq_archive.pkl'

print(f"📥 Downloading fresh 60d NQ 5m data... ({datetime.now().date()})")
df_new = yf.download("NQ=F", period="60d", interval="5m", progress=False)
df_new.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_new.columns]
df_new = df_new.dropna()
print(f"✅ Downloaded: {len(df_new)} bars ({df_new.index[0].date()} → {df_new.index[-1].date()})")

if os.path.exists(ARCHIVE):
    df_old = pd.read_pickle(ARCHIVE)
    print(f"📂 Existing archive: {len(df_old)} bars ({df_old.index[0].date()} → {df_old.index[-1].date()})")
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined = df_combined.sort_index()
else:
    df_combined = df_new
    print("📂 No existing archive — creating new one")

df_combined.to_pickle(ARCHIVE)
days = (df_combined.index[-1] - df_combined.index[0]).days
print(f"\n✅ Archive updated: {len(df_combined):,} bars spanning {days} days")
print(f"   Range: {df_combined.index[0].date()} → {df_combined.index[-1].date()}")
print(f"   File: {ARCHIVE}")
print(f"\n💡 Run this script weekly. Retrain RL agent when archive > 120 days.")
