"""
predict.py
Load model.pkl and evaluate a new trade setup.
Outputs: win probability, TAKE / SKIP recommendation,
and a breakdown of which factors are helping or hurting.
"""

import sys
import joblib
import numpy as np
import pandas as pd

# ── Load artefacts ────────────────────────────────────────────────────────────
try:
    artefacts      = joblib.load("model.pkl")
    model          = artefacts["model"]
    label_encoders = artefacts["label_encoders"]
    FEATURE_COLS   = artefacts["feature_cols"]
except FileNotFoundError:
    sys.exit("model.pkl not found — run train_model.py first.")

# ── Default example setup (edit or replace with live data) ───────────────────
DEFAULT_SETUP = {
    "timeframe":          "1H",
    "rsi_at_entry":       52.0,
    "ema_diff":           5.5,        # ema9 - ema21 in points
    "volume_ratio":       1.35,       # volume / vol_ma
    "session":            "london",
    "htf_bias":           "bullish",
    "trade_direction":    "long",
    "sl_distance_points": 18.0,
    "entry_price":        4850.0,
}


def engineer(raw: dict) -> pd.DataFrame:
    """Apply identical feature engineering as train_model.py."""
    d = raw.copy()

    # Derived flags
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

    # RSI zone
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

    # Volume tier
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

    # Encode categoricals
    cat_cols = ["timeframe", "session", "htf_bias", "trade_direction",
                "rsi_zone", "vol_tier"]
    for col in cat_cols:
        le = label_encoders[col]
        val = str(d[col])
        if val in le.classes_:
            d[col + "_enc"] = int(le.transform([val])[0])
        else:
            d[col + "_enc"] = 0   # fallback for unseen label

    row = pd.DataFrame([d])
    return row[FEATURE_COLS]


def factor_breakdown(raw: dict) -> list[tuple[str, str, str]]:
    """
    Return a list of (factor_name, assessment, detail) tuples
    categorised as GOOD / NEUTRAL / BAD.
    """
    factors = []
    d = raw

    # HTF bias alignment
    aligned = (
        (d["htf_bias"] == "bullish" and d["trade_direction"] == "long") or
        (d["htf_bias"] == "bearish" and d["trade_direction"] == "short")
    )
    if aligned:
        factors.append(("HTF Bias", "GOOD",
                         f"{d['htf_bias'].capitalize()} bias aligns with {d['trade_direction']} direction"))
    else:
        factors.append(("HTF Bias", "BAD",
                         f"{d['htf_bias'].capitalize()} bias conflicts with {d['trade_direction']} direction"))

    # RSI
    rsi = d["rsi_at_entry"]
    if 45 <= rsi <= 65:
        factors.append(("RSI", "GOOD", f"{rsi:.1f} — in the optimal 45-65 range"))
    elif 35 < rsi < 45 or 65 < rsi < 70:
        factors.append(("RSI", "NEUTRAL", f"{rsi:.1f} — acceptable but not ideal"))
    else:
        factors.append(("RSI", "BAD",
                         f"{rsi:.1f} — {'oversold extreme' if rsi <= 35 else 'overbought extreme'}"))

    # Volume
    vr = d["volume_ratio"]
    if vr > 1.5:
        factors.append(("Volume", "GOOD", f"{vr:.2f}x — strong confirmation"))
    elif vr > 1.2:
        factors.append(("Volume", "GOOD", f"{vr:.2f}x — above average"))
    elif vr > 0.8:
        factors.append(("Volume", "NEUTRAL", f"{vr:.2f}x — near average, weak confirmation"))
    else:
        factors.append(("Volume", "BAD", f"{vr:.2f}x — low volume, poor confirmation"))

    # Session
    sess = d["session"]
    if sess in ("london", "newyork"):
        factors.append(("Session", "GOOD", f"{sess.capitalize()} — high-liquidity session"))
    elif sess == "asia":
        factors.append(("Session", "NEUTRAL", f"Asia — moderate liquidity"))
    else:
        factors.append(("Session", "BAD", "Overnight — low liquidity, avoid"))

    # EMA alignment
    ema_d = d["ema_diff"]
    ema_bull = ema_d > 0
    direction = d["trade_direction"]
    if (direction == "long" and ema_bull) or (direction == "short" and not ema_bull):
        factors.append(("EMA Trend", "GOOD",
                         f"EMA diff {ema_d:+.1f} aligns with {direction}"))
    else:
        factors.append(("EMA Trend", "BAD",
                         f"EMA diff {ema_d:+.1f} conflicts with {direction}"))

    # Timeframe
    tf = d["timeframe"]
    if tf in ("4H", "Daily"):
        factors.append(("Timeframe", "GOOD", f"{tf} — stronger signal weight"))
    elif tf == "1H":
        factors.append(("Timeframe", "NEUTRAL", "1H — standard reliability"))
    else:
        factors.append(("Timeframe", "NEUTRAL", "15m — higher noise, tighter filter needed"))

    return factors


def predict(setup=None):
    raw = setup or DEFAULT_SETUP

    print("\n" + "=" * 58)
    print("  IFVG TRADE SETUP EVALUATOR")
    print("=" * 58)
    print(f"  Timeframe   : {raw['timeframe']}")
    print(f"  Direction   : {raw['trade_direction'].upper()}")
    print(f"  Session     : {raw['session'].capitalize()}")
    print(f"  HTF Bias    : {raw['htf_bias'].capitalize()}")
    print(f"  RSI         : {raw['rsi_at_entry']:.1f}")
    print(f"  EMA diff    : {raw['ema_diff']:+.2f} pts")
    print(f"  Volume ratio: {raw['volume_ratio']:.2f}x")
    print(f"  SL distance : {raw['sl_distance_points']:.1f} pts")
    print(f"  Entry price : {raw['entry_price']:.2f}")
    print("-" * 58)

    X = engineer(raw)
    prob_win  = float(model.predict_proba(X)[0, 1])
    prob_loss = 1.0 - prob_win

    # Recommendation thresholds
    if prob_win >= 0.62:
        rec   = "TAKE ✓"
        r_col = "STRONG"
    elif prob_win >= 0.52:
        rec   = "TAKE ✓"
        r_col = "MARGINAL"
    else:
        rec   = "SKIP ✗"
        r_col = ""

    print(f"\n  Win probability : {prob_win:.1%}")
    print(f"  Loss probability: {prob_loss:.1%}")

    bar_len  = 30
    filled   = round(prob_win * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] {prob_win:.0%}")

    print(f"\n  Recommendation  : {rec}  {r_col}")
    print("-" * 58)

    # Factor breakdown
    factors = factor_breakdown(raw)
    good    = [f for f in factors if f[1] == "GOOD"]
    neutral = [f for f in factors if f[1] == "NEUTRAL"]
    bad     = [f for f in factors if f[1] == "BAD"]

    if good:
        print("\n  ✅  HELPING:")
        for name, _, detail in good:
            print(f"      {name:<14} {detail}")

    if neutral:
        print("\n  ⚠️   NEUTRAL:")
        for name, _, detail in neutral:
            print(f"      {name:<14} {detail}")

    if bad:
        print("\n  ❌  HURTING:")
        for name, _, detail in bad:
            print(f"      {name:<14} {detail}")

    print("=" * 58 + "\n")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # To evaluate a custom setup, replace DEFAULT_SETUP above
    # or call predict({...}) with your own dict.
    predict()
