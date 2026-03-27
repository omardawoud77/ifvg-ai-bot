"""
predict.py
Load model.pkl and evaluate a new trade setup.
Outputs: win probability, TAKE / SKIP recommendation,
and a breakdown of which factors are helping or hurting.
Trained on real NQ/MNQ trade data.
"""

import sys
import joblib
import numpy as np
import pandas as pd

# ── Load model ────────────────────────────────────────────────────────────────
try:
    artefacts      = joblib.load("model.pkl")
    model          = artefacts["model"]
    label_encoders = artefacts["label_encoders"]
    FEATURE_COLS   = artefacts["feature_cols"]
except FileNotFoundError:
    sys.exit("model.pkl not found — run train_real.py first.")

# ── Edit this setup before running ───────────────────────────────────────────
DEFAULT_SETUP = {
    "timeframe":          "1H",
    "rsi_at_entry":       52.0,      # RSI value at your entry bar
    "ema_diff":           6.0,       # EMA9 minus EMA21 in points
    "volume_ratio":       1.4,       # current volume / 20-bar avg volume
    "session":            "newyork", # london / newyork / asia / overnight
    "htf_bias":           "bullish", # bullish / bearish
    "trade_direction":    "long",    # long / short
    "sl_distance_points": 20.0,      # distance from entry to your SL in points
    "entry_price":        21000.0,   # your entry price
}


def engineer(raw: dict) -> pd.DataFrame:
    d = raw.copy()

    d["bias_aligned"] = int(
        (d["htf_bias"] == "bullish" and d["trade_direction"] == "long") or
        (d["htf_bias"] == "bearish" and d["trade_direction"] == "short")
    )
    d["ema_aligned"] = int(
        (d["ema_diff"] > 0 and d["trade_direction"] == "long") or
        (d["ema_diff"] < 0 and d["trade_direction"] == "short")
    )

    session_quality = {"london": 2, "newyork": 2, "asia": 1, "overnight": 0}
    d["session_quality"] = session_quality.get(d["session"], 0)
    d["bias_x_session"]  = d["bias_aligned"] * d["session_quality"]
    d["rsi_dist_50"]     = abs(d["rsi_at_entry"] - 50)
    d["ema_abs"]         = abs(d["ema_diff"])

    rsi = d["rsi_at_entry"]
    if rsi <= 35:
        d["rsi_zone"] = "oversold"
    elif rsi <= 45:
        d["rsi_zone"] = "low_neutral"
    elif rsi <= 55:
        d["rsi_zone"] = "mid_neutral"
    elif rsi <= 65:
        d["rsi_zone"] = "high_neutral"
    else:
        d["rsi_zone"] = "overbought"

    vr = d["volume_ratio"]
    if vr <= 0.8:
        d["vol_tier"] = "very_low"
    elif vr <= 1.0:
        d["vol_tier"] = "low"
    elif vr <= 1.2:
        d["vol_tier"] = "normal"
    elif vr <= 1.5:
        d["vol_tier"] = "high"
    else:
        d["vol_tier"] = "very_high"

    cat_cols = ["timeframe", "session", "htf_bias", "trade_direction",
                "rsi_zone", "vol_tier"]
    for col in cat_cols:
        le = label_encoders[col]
        val = str(d[col])
        if val in le.classes_:
            d[col + "_enc"] = int(le.transform([val])[0])
        else:
            d[col + "_enc"] = 0

    row = pd.DataFrame([d])
    return row[FEATURE_COLS]


def factor_breakdown(raw: dict) -> list:
    factors = []
    d = raw

    # HTF bias alignment
    aligned = (
        (d["htf_bias"] == "bullish" and d["trade_direction"] == "long") or
        (d["htf_bias"] == "bearish" and d["trade_direction"] == "short")
    )
    if aligned:
        factors.append(("HTF Bias", "GOOD",
            f"{d['htf_bias'].capitalize()} bias aligns with {d['trade_direction']}"))
    else:
        factors.append(("HTF Bias", "BAD",
            f"{d['htf_bias'].capitalize()} bias conflicts with {d['trade_direction']}"))

    # RSI
    rsi = d["rsi_at_entry"]
    if 45 <= rsi <= 65:
        factors.append(("RSI", "GOOD", f"{rsi:.1f} — optimal 45-65 range"))
    elif 35 < rsi < 45 or 65 < rsi < 70:
        factors.append(("RSI", "NEUTRAL", f"{rsi:.1f} — acceptable but not ideal"))
    else:
        factors.append(("RSI", "BAD",
            f"{rsi:.1f} — {'oversold extreme' if rsi <= 35 else 'overbought extreme'}"))

    # EMA momentum (top feature in real model)
    ema_d = d["ema_diff"]
    ema_bull = ema_d > 0
    direction = d["trade_direction"]
    ema_mag = abs(ema_d)
    if (direction == "long" and ema_bull) or (direction == "short" and not ema_bull):
        if ema_mag > 10:
            factors.append(("EMA Trend", "GOOD",
                f"Strong momentum {ema_d:+.1f} pts aligns with {direction}"))
        else:
            factors.append(("EMA Trend", "GOOD",
                f"EMA diff {ema_d:+.1f} aligns with {direction}"))
    else:
        factors.append(("EMA Trend", "BAD",
            f"EMA diff {ema_d:+.1f} conflicts with {direction}"))

    # Volume
    vr = d["volume_ratio"]
    if vr > 1.5:
        factors.append(("Volume", "GOOD", f"{vr:.2f}x — strong confirmation"))
    elif vr > 1.2:
        factors.append(("Volume", "GOOD", f"{vr:.2f}x — above average"))
    elif vr > 0.8:
        factors.append(("Volume", "NEUTRAL", f"{vr:.2f}x — weak confirmation"))
    else:
        factors.append(("Volume", "BAD", f"{vr:.2f}x — low volume, poor confirmation"))

    # Session
    sess = d["session"]
    if sess in ("london", "newyork"):
        factors.append(("Session", "GOOD", f"{sess.capitalize()} — high liquidity"))
    elif sess == "asia":
        factors.append(("Session", "NEUTRAL", "Asia — moderate liquidity"))
    else:
        factors.append(("Session", "BAD", "Overnight — avoid"))

    # SL distance
    sl = d["sl_distance_points"]
    if sl <= 15:
        factors.append(("SL Distance", "GOOD", f"{sl:.1f} pts — tight, good R:R potential"))
    elif sl <= 30:
        factors.append(("SL Distance", "NEUTRAL", f"{sl:.1f} pts — acceptable"))
    else:
        factors.append(("SL Distance", "BAD", f"{sl:.1f} pts — wide SL, poor R:R"))

    return factors


def predict(setup=None):
    raw = setup or DEFAULT_SETUP

    print("\n" + "=" * 60)
    print("  IFVG TRADE EVALUATOR  —  NQ / MNQ")
    print("=" * 60)
    print(f"  Direction   : {raw['trade_direction'].upper()}")
    print(f"  Session     : {raw['session'].capitalize()}")
    print(f"  HTF Bias    : {raw['htf_bias'].capitalize()}")
    print(f"  RSI         : {raw['rsi_at_entry']:.1f}")
    print(f"  EMA diff    : {raw['ema_diff']:+.2f} pts")
    print(f"  Volume ratio: {raw['volume_ratio']:.2f}x")
    print(f"  SL distance : {raw['sl_distance_points']:.1f} pts")
    print(f"  Entry price : {raw['entry_price']:.2f}")
    print("-" * 60)

    X = engineer(raw)
    prob_win  = float(model.predict_proba(X)[0, 1])
    prob_loss = 1.0 - prob_win

    # Recommendation thresholds
    if prob_win >= 0.70:
        rec   = "✅  TAKE  —  STRONG"
    elif prob_win >= 0.58:
        rec   = "✅  TAKE  —  MARGINAL"
    else:
        rec   = "❌  SKIP"

    print(f"\n  Win probability : {prob_win:.1%}")
    print(f"  Loss probability: {prob_loss:.1%}")

    bar_len = 30
    filled  = round(prob_win * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] {prob_win:.0%}")
    print(f"\n  Recommendation  : {rec}")
    print("-" * 60)

    factors = factor_breakdown(raw)
    good    = [f for f in factors if f[1] == "GOOD"]
    neutral = [f for f in factors if f[1] == "NEUTRAL"]
    bad     = [f for f in factors if f[1] == "BAD"]

    if good:
        print("\n  ✅  HELPING:")
        for name, _, detail in good:
            print(f"      {name:<15} {detail}")
    if neutral:
        print("\n  ⚠️   NEUTRAL:")
        for name, _, detail in neutral:
            print(f"      {name:<15} {detail}")
    if bad:
        print("\n  ❌  HURTING:")
        for name, _, detail in bad:
            print(f"      {name:<15} {detail}")

    print("=" * 60 + "\n")
    return prob_win


if __name__ == "__main__":
    predict()
