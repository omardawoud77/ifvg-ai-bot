"""
live_app.py — IFVG Live AI v7
Full MTF ICT Framework + Auto Paper Trading + AI Learning

ICT Layer (NEW in v7):
  - Full multi-timeframe analysis: Weekly → Daily → 4H → 1H → 15m → 5m
  - FVG detection on all timeframes
  - Order block identification (Daily, 4H, 1H)
  - Liquidity sweep detection (equal highs/lows)
  - Market structure (BOS/CHoCH) tracking
  - Trade type: Continuation vs Reversal with separate scoring weights
  - Absorption candle detection on 5m
  - Kill zone precision (London open, NY open, NY PM)
  - ICT score layer added on top of XGBoost base score
"""

from flask import Flask, jsonify, render_template_string, request as freq
import joblib, numpy as np, pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys, json, os, time, threading, requests as req

app = Flask(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
try:
    art   = joblib.load("model.pkl")
    model = art["model"]
    les   = art["label_encoders"]
    FCOLS = art["feature_cols"]
    print("✅ Model loaded")
except Exception as e:
    sys.exit(f"model.pkl not found — run train_real.py first ({e})")

TRADE_LOG       = "live_trades.json"
RR              = 2.0
MNQ_PTS_TO_USD  = 2.0
SCORE_THRESHOLD = 78

last_fetch   = 0
INTERVAL     = 15
last_htf_fetch = 0
HTF_INTERVAL = 900  # 15 min for HTF  # fetch weekly/daily/4H every 10 minutes only
cached_htf = {}
retrain_lock = threading.Lock()

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "score": 0, "prob": 0.0, "take": False, "direction": "—",
    "score_long": 0, "score_short": 0,
    "base_score": 0, "ict_score": 0,
    "rsi": 50.0, "ema_diff": 0.0, "vol_ratio": 1.0,
    "session": "—", "htf_bias": "—", "sl_dist": 0.0,
    "price": 0.0, "sl_price": 0.0, "tp_price": 0.0,
    "last_update": "—", "error": None, "factors": [],
    "ict_factors": [],
    "trade_type": "—",
    "alert": False, "alert_msg": "",
    "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
    "total_pnl_usd": 0.0,
    "active_trade": None,
    "trades": [], "stats": {},
    "model_version": 0, "last_retrain": "—",
    "data_source": "Yahoo Finance",
    "mtf": {},
}

prev_score = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_trades():
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG) as f: return json.load(f)
    return []

def save_trades(t):
    with open(TRADE_LOG, "w") as f: json.dump(t, f, indent=2)

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(com=p-1, adjust=False).mean()
    al = l.ewm(com=p-1, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

# ── Price fetch (multi-timeframe) ─────────────────────────────────────────────
TF_MAP = {
    "weekly":  ("1wk", "1y"),
    "daily":   ("1d",  "6mo"),
    "4h":      ("1h",  "60d"),   # Yahoo doesn't have 4h; we resample from 1h
    "1h":      ("1h",  "30d"),
    "15m":     ("15m", "5d"),
    "5m":      ("5m",  "2d"),
}

_yf_session = None
_yf_crumb = None

def get_yf_crumb():
    global _yf_session, _yf_crumb
    if _yf_crumb:
        return _yf_session, _yf_crumb
    try:
        _yf_session = req.Session()
        _yf_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
        })
        _yf_session.get("https://fc.yahoo.com", timeout=5)
        r = _yf_session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=5)
        _yf_crumb = r.text.strip()
        print(f"✅ Yahoo crumb obtained: {_yf_crumb[:8]}...")
        return _yf_session, _yf_crumb
    except Exception as e:
        print(f"Crumb fetch failed: {e}")
        _yf_session = None
        _yf_crumb = None
        return None, None

def fetch_tf(interval, range_str, n_bars=100):
    global _yf_crumb, _yf_session
    for attempt in range(2):
        for host in ["query2", "query1"]:
            try:
                url = f"https://{host}.finance.yahoo.com/v8/finance/chart/NQ=F?interval={interval}&range={range_str}"
                if _yf_crumb:
                    url += f"&crumb={_yf_crumb}"
                hdrs = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Accept": "application/json"}
                r = (_yf_session or req).get(url, headers=hdrs, timeout=12)
                if r.status_code == 429:
                    print(f"429 on {host}")
                    _yf_crumb = None
                    _yf_session = None
                    continue
                if r.status_code != 200:
                    continue
                js = r.json()
                res = js["chart"]["result"][0]
                if "timestamp" not in res:
                    continue
                ts = res["timestamp"]
                q = res["indicators"]["quote"][0]
                bars = pd.DataFrame({
                    "Open": q["open"], "High": q["high"],
                    "Low": q["low"], "Close": q["close"],
                    "Volume": q["volume"],
                }, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()
                return bars.tail(n_bars) if len(bars) >= 5 else None
            except Exception as e:
                print(f"fetch_tf error ({interval}/{host}): {e}")
        if attempt == 0:
            get_yf_crumb()
    return None

def fetch_all_timeframes():
    global last_htf_fetch, cached_htf
    now = time.time()
    bars = {}

    # ── Fast timeframes: 5m every cycle, 15m every cycle ──────────────────────
    bars["5m"]  = fetch_tf("5m",  "2d",  100)
    time.sleep(2)
    bars["15m"] = fetch_tf("15m", "5d",  100)
    time.sleep(2)

    # ── Slow timeframes: only refresh every HTF_INTERVAL ──────────────────────
    if now - last_htf_fetch > HTF_INTERVAL or not cached_htf:
        time.sleep(2)
        bars["1h"]     = fetch_tf("1h",  "30d", 200)
        time.sleep(2)
        bars["daily"]  = fetch_tf("1d",  "6mo", 120)
        time.sleep(2)
        bars["weekly"] = fetch_tf("1wk", "2y",  52)
        time.sleep(2)

        if bars["1h"] is not None and len(bars["1h"]) >= 4:
            try:
                bars["4h"] = bars["1h"].resample("4h").agg({
                    "Open": "first", "High": "max",
                    "Low": "min", "Close": "last", "Volume": "sum"
                }).dropna().tail(60)
            except:
                bars["4h"] = None
        else:
            bars["4h"] = None

        cached_htf = {k: bars[k] for k in ["weekly","daily","1h","4h"] if k in bars}
        last_htf_fetch = now
        print("📊 HTF data refreshed via Yahoo Finance")
    else:
        bars.update(cached_htf)

    return bars

# ── ICT Concepts ──────────────────────────────────────────────────────────────

def detect_fvg(bars, direction=None, lookback=30):
    """
    Detect Fair Value Gaps in last N bars.
    FVG bullish: bar[i].low > bar[i-2].high  (gap up — unfilled space)
    FVG bearish: bar[i].high < bar[i-2].low  (gap down — unfilled space)
    Returns list of dicts with type, top, bottom, midpoint, index, filled
    """
    if bars is None or len(bars) < 3:
        return []
    fvgs = []
    df = bars.tail(lookback).reset_index(drop=True)
    price_now = df["Close"].iloc[-1]

    for i in range(2, len(df)):
        high_prev2 = df["High"].iloc[i-2]
        low_prev2  = df["Low"].iloc[i-2]
        high_curr  = df["High"].iloc[i]
        low_curr   = df["Low"].iloc[i]

        # Bullish FVG: current bar's low > 2 bars ago's high
        if low_curr > high_prev2:
            gap_top    = low_curr
            gap_bottom = high_prev2
            midpoint   = (gap_top + gap_bottom) / 2
            filled     = price_now <= gap_top and price_now >= gap_bottom
            touching   = abs(price_now - midpoint) / midpoint < 0.002  # within 0.2%
            if direction is None or direction == "bullish":
                fvgs.append({
                    "type": "bullish", "top": round(gap_top, 2),
                    "bottom": round(gap_bottom, 2), "mid": round(midpoint, 2),
                    "filled": filled, "touching": touching, "bar_idx": i,
                    "age": len(df) - i,
                })

        # Bearish FVG: current bar's high < 2 bars ago's low
        if high_curr < low_prev2:
            gap_top    = low_prev2
            gap_bottom = high_curr
            midpoint   = (gap_top + gap_bottom) / 2
            filled     = price_now <= gap_top and price_now >= gap_bottom
            touching   = abs(price_now - midpoint) / midpoint < 0.002
            if direction is None or direction == "bearish":
                fvgs.append({
                    "type": "bearish", "top": round(gap_top, 2),
                    "bottom": round(gap_bottom, 2), "mid": round(midpoint, 2),
                    "filled": filled, "touching": touching, "bar_idx": i,
                    "age": len(df) - i,
                })

    return fvgs

def detect_order_blocks(bars, lookback=50):
    """
    Order block: last significant up/down candle before a strong move away.
    Bullish OB: last bearish candle before strong bullish move
    Bearish OB: last bullish candle before strong bearish move
    """
    if bars is None or len(bars) < 10:
        return []
    obs = []
    df = bars.tail(lookback).reset_index(drop=True)
    price_now = df["Close"].iloc[-1]
    atr = calc_atr(df["High"], df["Low"], df["Close"]).iloc[-1]

    for i in range(1, len(df) - 2):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        c2 = df.iloc[i+2] if i+2 < len(df) else None

        # Bullish OB: bearish candle followed by strong bullish move
        if c0["Close"] < c0["Open"]:  # bearish candle
            move = c1["Close"] - c0["Close"]
            if move > atr * 1.5:  # strong bullish move after
                near = abs(price_now - c0["Low"]) < atr * 2
                obs.append({
                    "type": "bullish",
                    "top": round(c0["Open"], 2),
                    "bottom": round(c0["Low"], 2),
                    "mid": round((c0["Open"] + c0["Low"]) / 2, 2),
                    "near": near, "age": len(df) - i,
                })

        # Bearish OB: bullish candle followed by strong bearish move
        if c0["Close"] > c0["Open"]:  # bullish candle
            move = c0["Close"] - c1["Close"]
            if move > atr * 1.5:  # strong bearish move after
                near = abs(price_now - c0["High"]) < atr * 2
                obs.append({
                    "type": "bearish",
                    "top": round(c0["High"], 2),
                    "bottom": round(c0["Close"], 2),
                    "mid": round((c0["High"] + c0["Close"]) / 2, 2),
                    "near": near, "age": len(df) - i,
                })

    return obs[-5:] if obs else []  # last 5 order blocks

def detect_liquidity_sweep(bars, lookback=40):
    """
    Liquidity sweep: price takes out equal highs or equal lows (within 0.1%)
    then quickly reverses. This is a stop hunt.
    Returns: swept_high, swept_low, sweep_type, reclaimed
    """
    if bars is None or len(bars) < 15:
        return {"swept": False, "type": None, "level": None, "reclaimed": False}

    df = bars.tail(lookback).reset_index(drop=True)
    price_now = df["Close"].iloc[-1]
    atr = calc_atr(df["High"], df["Low"], df["Close"]).iloc[-1]

    # Find equal highs (within 0.15% of each other) in last 20 bars
    recent = df.iloc[-20:-1]
    highs = recent["High"].values
    lows  = recent["Low"].values

    # Check if last candle swept above a previous high then closed below
    last_high  = df["High"].iloc[-1]
    last_close = df["Close"].iloc[-1]
    last_open  = df["Open"].iloc[-1]

    # Sweep of highs: wick above recent swing high, closed back below
    swing_high = recent["High"].max()
    if last_high > swing_high and last_close < swing_high:
        return {
            "swept": True, "type": "high_sweep",
            "level": round(swing_high, 2),
            "reclaimed": True,
            "extreme": last_high - swing_high > atr * 0.5,
        }

    # Sweep of lows: wick below recent swing low, closed back above
    swing_low = recent["Low"].min()
    last_low = df["Low"].iloc[-1]
    if last_low < swing_low and last_close > swing_low:
        return {
            "swept": True, "type": "low_sweep",
            "level": round(swing_low, 2),
            "reclaimed": True,
            "extreme": swing_low - last_low > atr * 0.5,
        }

    return {"swept": False, "type": None, "level": None, "reclaimed": False, "extreme": False}

def detect_market_structure(bars, lookback=50):
    """
    Market structure: track swing highs/lows to determine BOS/CHoCH.
    Bullish structure: series of HH + HL
    Bearish structure: series of LH + LL
    Returns: structure (bullish/bearish/ranging), last_bos, choch
    """
    if bars is None or len(bars) < 20:
        return {"structure": "ranging", "bos": None, "choch": False, "swing_high": None, "swing_low": None}

    df = bars.tail(lookback).reset_index(drop=True)

    # Find swing highs and lows (simple: local max/min over 5 bars)
    highs = []
    lows  = []
    for i in range(2, len(df) - 2):
        if df["High"].iloc[i] == df["High"].iloc[i-2:i+3].max():
            highs.append((i, df["High"].iloc[i]))
        if df["Low"].iloc[i] == df["Low"].iloc[i-2:i+3].min():
            lows.append((i, df["Low"].iloc[i]))

    if len(highs) < 2 or len(lows) < 2:
        return {"structure": "ranging", "bos": None, "choch": False,
                "swing_high": None, "swing_low": None}

    # Last 2 swing highs and lows
    last_highs = highs[-2:]
    last_lows  = lows[-2:]

    hh = last_highs[-1][1] > last_highs[-2][1]  # higher high
    hl = last_lows[-1][1]  > last_lows[-2][1]   # higher low
    lh = last_highs[-1][1] < last_highs[-2][1]  # lower high
    ll = last_lows[-1][1]  < last_lows[-2][1]   # lower low

    if hh and hl:
        structure = "bullish"
    elif lh and ll:
        structure = "bearish"
    else:
        structure = "ranging"

    # BOS: price breaks above last swing high (bullish BOS) or below last swing low (bearish BOS)
    price_now = df["Close"].iloc[-1]
    bos = None
    if price_now > last_highs[-1][1]:
        bos = "bullish_bos"
    elif price_now < last_lows[-1][1]:
        bos = "bearish_bos"

    # CHoCH: structure changes (was bullish, now bearish signal or vice versa)
    choch = (structure == "bullish" and ll) or (structure == "bearish" and hh)

    return {
        "structure": structure, "bos": bos, "choch": choch,
        "swing_high": round(last_highs[-1][1], 2),
        "swing_low":  round(last_lows[-1][1], 2),
    }

def detect_absorption(bars_5m, lookback=10):
    """
    Absorption: large volume candle with small body at a key level.
    Big bid: bullish absorption (down candle with huge volume, closes near open)
    Big offer: bearish absorption
    """
    if bars_5m is None or len(bars_5m) < 5:
        return {"absorbed": False, "type": None}

    df = bars_5m.tail(lookback).reset_index(drop=True)
    vol_avg = df["Volume"].mean()
    last = df.iloc[-1]

    body_pct = abs(last["Close"] - last["Open"]) / (last["High"] - last["Low"] + 0.001)
    high_vol  = last["Volume"] > vol_avg * 1.5

    if high_vol and body_pct < 0.35:
        # Small body + high volume = absorption
        direction = "bullish" if last["Close"] >= last["Open"] else "bearish"
        return {"absorbed": True, "type": direction, "vol_ratio": round(last["Volume"] / vol_avg, 1)}

    return {"absorbed": False, "type": None, "vol_ratio": 0}

def get_kill_zone():
    """More precise kill zone detection than basic session."""
    now  = datetime.now(timezone.utc)
    hour = now.hour
    mins = now.hour * 60 + now.minute

    # London open: 2:00-5:00 AM UTC
    if 2 <= hour < 5:
        return "london_open", True
    # NY open: 9:30-11:00 AM UTC (14:30-16:00 UTC)
    if 870 <= mins < 960:
        return "ny_open", True
    # London close / NY overlap: 3:00-4:00 PM UTC
    if 900 <= mins < 960:
        return "london_close", True
    # NY PM session: 1:30-3:00 PM UTC
    if 810 <= mins < 900:
        return "ny_pm", True
    # Asia: 8 PM - 2 AM UTC
    if hour >= 20 or hour < 2:
        return "asia", False

    return "transition", False

# ── Full MTF ICT Analysis ─────────────────────────────────────────────────────

def run_ict_analysis(all_bars, direction, price_now):
    """
    Run full ICT analysis across all timeframes.
    Returns: ict_score (int), ict_factors (list), trade_type (str), mtf_summary (dict)
    """
    ict_score  = 0
    factors    = []
    trade_type = "none"

    # ── 1. Weekly structure ───────────────────────────────────────────────────
    w_struct = detect_market_structure(all_bars.get("weekly"), lookback=30)
    w_fvgs   = detect_fvg(all_bars.get("weekly"), lookback=20)

    w_bullish = w_struct["structure"] == "bullish"
    w_bearish = w_struct["structure"] == "bearish"

    w_fvg_target = None
    for fvg in w_fvgs:
        if not fvg["filled"]:
            w_fvg_target = fvg
            break  # closest unfilled FVG

    factors.append({
        "name": "Weekly Structure",
        "value": w_struct["structure"].upper(),
        "aligned": (w_bullish and direction == "long") or (w_bearish and direction == "short"),
        "detail": f"{'HH/HL' if w_bullish else 'LH/LL' if w_bearish else 'Ranging'}",
    })

    # ── 2. Daily structure + FVG targets ─────────────────────────────────────
    d_struct = detect_market_structure(all_bars.get("daily"), lookback=60)
    d_fvgs   = detect_fvg(all_bars.get("daily"), lookback=40)
    d_obs    = detect_order_blocks(all_bars.get("daily"), lookback=60)

    d_bullish = d_struct["structure"] == "bullish"
    d_bearish = d_struct["structure"] == "bearish"

    # Find nearest unfilled daily FVG as target
    d_fvg_target = None
    for fvg in sorted(d_fvgs, key=lambda x: abs(x["mid"] - price_now)):
        if not fvg["filled"]:
            d_fvg_target = fvg
            break

    # Check if price is near a daily order block
    d_ob_near = any(ob["near"] and ob["type"] == ("bullish" if direction=="long" else "bearish")
                    for ob in d_obs)

    factors.append({
        "name": "Daily Structure",
        "value": d_struct["structure"].upper(),
        "aligned": (d_bullish and direction == "long") or (d_bearish and direction == "short"),
        "detail": f"Target FVG: {d_fvg_target['mid'] if d_fvg_target else 'None'}",
    })

    # ── 3. 4H structure ───────────────────────────────────────────────────────
    h4_struct = detect_market_structure(all_bars.get("4h"), lookback=40)
    h4_fvgs   = detect_fvg(all_bars.get("4h"), lookback=30)

    h4_bullish = h4_struct["structure"] == "bullish"
    h4_bearish = h4_struct["structure"] == "bearish"

    # FVG respect: are we respecting bullish or bearish FVGs on 4H?
    h4_bullish_fvgs_respected = any(
        not fvg["filled"] and fvg["type"] == "bullish" for fvg in h4_fvgs
    )
    h4_bearish_fvgs_respected = any(
        not fvg["filled"] and fvg["type"] == "bearish" for fvg in h4_fvgs
    )

    factors.append({
        "name": "4H Structure",
        "value": h4_struct["structure"].upper(),
        "aligned": (h4_bullish and direction == "long") or (h4_bearish and direction == "short"),
        "detail": f"BOS: {h4_struct['bos'] or 'None'}",
    })

    # ── 4. 1H structure + FVG respect/disrespect ─────────────────────────────
    h1_struct = detect_market_structure(all_bars.get("1h"), lookback=50)
    h1_fvgs   = detect_fvg(all_bars.get("1h"), lookback=30)

    h1_bullish = h1_struct["structure"] == "bullish"
    h1_bearish = h1_struct["structure"] == "bearish"

    # Key: are we respecting or disrespecting FVGs?
    # Bearish structure = disrespecting bullish FVGs, respecting bearish FVGs
    h1_bullish_fvg_touching = any(fvg["touching"] and fvg["type"] == "bullish" for fvg in h1_fvgs)
    h1_bearish_fvg_touching = any(fvg["touching"] and fvg["type"] == "bearish" for fvg in h1_fvgs)

    factors.append({
        "name": "1H Structure",
        "value": h1_struct["structure"].upper(),
        "aligned": (h1_bullish and direction == "long") or (h1_bearish and direction == "short"),
        "detail": f"CHoCH: {'Yes' if h1_struct['choch'] else 'No'}",
    })

    # ── 5. 15m structure + liquidity sweep ───────────────────────────────────
    m15_struct = detect_market_structure(all_bars.get("15m"), lookback=40)
    m15_sweep  = detect_liquidity_sweep(all_bars.get("15m"), lookback=30)

    m15_bullish = m15_struct["structure"] == "bullish"
    m15_bearish = m15_struct["structure"] == "bearish"

    factors.append({
        "name": "15m Sweep",
        "value": m15_sweep["type"].upper().replace("_", " ") if m15_sweep["swept"] else "NONE",
        "aligned": m15_sweep["swept"],
        "detail": f"Level: {m15_sweep['level'] or '—'} Extreme: {'Yes' if m15_sweep.get('extreme') else 'No'}",
    })

    # ── 6. 5m FVG + absorption + IFVG ────────────────────────────────────────
    m5_fvgs  = detect_fvg(all_bars.get("5m"), lookback=30)
    m5_abs   = detect_absorption(all_bars.get("5m"), lookback=10)
    m5_sweep = detect_liquidity_sweep(all_bars.get("5m"), lookback=20)

    m5_fvg_touch = any(
        fvg["touching"] and fvg["type"] == ("bullish" if direction=="long" else "bearish")
        for fvg in m5_fvgs
    )

    # IFVG: price is in an inverted FVG (previously bearish FVG now acting as support)
    m5_ifvg = any(
        fvg["touching"] and fvg["type"] != ("bullish" if direction=="long" else "bearish")
        for fvg in m5_fvgs
    )

    factors.append({
        "name": "5m FVG",
        "value": "TOUCHING" if m5_fvg_touch else "NOT TOUCHING",
        "aligned": m5_fvg_touch,
        "detail": f"Absorption: {'Yes ' + m5_abs['type'] if m5_abs['absorbed'] else 'No'}",
    })

    # ── Kill zone ─────────────────────────────────────────────────────────────
    kz_name, kz_active = get_kill_zone()
    factors.append({
        "name": "Kill Zone",
        "value": kz_name.upper().replace("_", " "),
        "aligned": kz_active,
        "detail": "High probability window" if kz_active else "Outside kill zone",
    })

    # ── Liquidity direction (external → internal) ─────────────────────────────
    # Price should be moving from external liquidity (swing H/L) toward internal (FVG)
    d_swing_high = d_struct.get("swing_high")
    d_swing_low  = d_struct.get("swing_low")

    moving_to_target = False
    if direction == "long" and d_swing_high and price_now < d_swing_high:
        moving_to_target = True  # bullish, moving toward swing high (external)
    elif direction == "short" and d_swing_low and price_now > d_swing_low:
        moving_to_target = True  # bearish, moving toward swing low (external)

    factors.append({
        "name": "Liquidity Draw",
        "value": "ALIGNED" if moving_to_target else "NO TARGET",
        "aligned": moving_to_target,
        "detail": f"Target: {'High' if direction=='long' else 'Low'} @ {d_swing_high if direction=='long' else d_swing_low}",
    })

    # ── SCORING ───────────────────────────────────────────────────────────────

    # Count aligned timeframes
    struct_alignment = sum([
        (w_bullish and direction=="long") or (w_bearish and direction=="short"),
        (d_bullish and direction=="long") or (d_bearish and direction=="short"),
        (h4_bullish and direction=="long") or (h4_bearish and direction=="short"),
        (h1_bullish and direction=="long") or (h1_bearish and direction=="short"),
        (m15_bullish and direction=="long") or (m15_bearish and direction=="short"),
    ])

    # HTF agreement (weekly + daily)
    htf_agree = (
        (w_bullish and d_bullish and direction=="long") or
        (w_bearish and d_bearish and direction=="short")
    )

    if htf_agree:             ict_score += 15
    if struct_alignment >= 4: ict_score += 10
    elif struct_alignment == 3: ict_score += 5
    elif struct_alignment <= 1: ict_score -= 20  # strong conflict

    if d_fvg_target:          ict_score += 10   # daily FVG target identified
    if d_ob_near:             ict_score += 8    # near daily OB
    if moving_to_target:      ict_score += 10   # moving toward liquidity
    if m5_fvg_touch:          ict_score += 15   # 5m FVG entry zone
    if m5_abs["absorbed"]:    ict_score += 15   # absorption candle
    if m5_ifvg:               ict_score += 10   # IFVG entry
    if kz_active:             ict_score += 5    # kill zone

    # Against HTF bias — strong penalty
    htf_against = (
        (w_bearish and direction=="long") or
        (w_bullish and direction=="short")
    )
    if htf_against:           ict_score -= 25

    # CHoCH on 1H is good for reversals
    if h1_struct["choch"]:    ict_score += 8

    # ── Trade type detection ──────────────────────────────────────────────────
    is_continuation = (
        struct_alignment >= 3 and
        m5_fvg_touch and
        m5_abs["absorbed"] and
        moving_to_target
    )

    extreme_sweep = (
        m15_sweep["swept"] and
        m15_sweep.get("extreme", False) and
        h1_struct["choch"]
    )

    is_reversal = (
        extreme_sweep and
        struct_alignment <= 2 and
        m5_fvg_touch
    )

    if is_continuation and is_reversal:
        trade_type = "continuation+reversal"
        ict_score += 5  # both signals = extra confirmation
    elif is_continuation:
        trade_type = "continuation"
    elif is_reversal:
        trade_type = "reversal"
        ict_score += 10  # reversal bonus (harder to get, more reliable)
    else:
        trade_type = "no_setup"
        ict_score -= 10  # no clear ICT setup

    # Cap ICT score
    ict_score = max(-30, min(30, ict_score))

    mtf_summary = {
        "weekly":  w_struct["structure"],
        "daily":   d_struct["structure"],
        "4h":      h4_struct["structure"],
        "1h":      h1_struct["structure"],
        "15m":     m15_struct["structure"],
        "kz":      kz_name,
        "kz_active": kz_active,
        "d_fvg_target": d_fvg_target["mid"] if d_fvg_target else None,
        "sweep":   m15_sweep,
        "absorption": m5_abs,
        "alignment": struct_alignment,
        "htf_agree": htf_agree,
    }

    return ict_score, factors, trade_type, mtf_summary

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer(raw):
    d = raw.copy()
    d["bias_aligned"]    = int((d["htf_bias"]=="bullish" and d["trade_direction"]=="long") or
                               (d["htf_bias"]=="bearish" and d["trade_direction"]=="short"))
    d["ema_aligned"]     = int((d["ema_diff"]>0 and d["trade_direction"]=="long") or
                               (d["ema_diff"]<0 and d["trade_direction"]=="short"))
    sq                   = {"london":2,"newyork":2,"asia":1,"overnight":0}
    d["session_quality"] = sq.get(d["session"], 0)
    d["bias_x_session"]  = d["bias_aligned"] * d["session_quality"]
    d["rsi_dist_50"]     = abs(d["rsi_at_entry"] - 50)
    d["ema_abs"]         = abs(d["ema_diff"])
    r = d["rsi_at_entry"]
    d["rsi_zone"] = ("oversold" if r<=35 else "low_neutral" if r<=45
                     else "mid_neutral" if r<=55 else "high_neutral" if r<=65 else "overbought")
    v = d["volume_ratio"]
    d["vol_tier"] = ("very_low" if v<=0.8 else "low" if v<=1.0
                     else "normal" if v<=1.2 else "high" if v<=1.5 else "very_high")
    for col in ["timeframe","session","htf_bias","trade_direction","rsi_zone","vol_tier"]:
        le = les[col]; val = str(d[col])
        d[col+"_enc"] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([d])[FCOLS]

def get_base_factors(raw):
    f  = []
    al = ((raw["htf_bias"]=="bullish" and raw["trade_direction"]=="long") or
          (raw["htf_bias"]=="bearish" and raw["trade_direction"]=="short"))
    f.append({"name":"HTF Bias","rating":"good" if al else "bad",
               "detail":f"{'Aligns' if al else 'Conflicts'} with {raw['trade_direction']}"})
    r = raw["rsi_at_entry"]
    f.append({"name":"RSI","rating":"good" if 45<=r<=65 else "neutral" if 35<r<70 else "bad","detail":f"{r:.1f}"})
    ed = raw["ema_diff"]
    eo = (ed>0 and raw["trade_direction"]=="long") or (ed<0 and raw["trade_direction"]=="short")
    f.append({"name":"EMA","rating":"good" if eo else "bad","detail":f"{ed:+.1f}pts"})
    vr = raw["volume_ratio"]
    f.append({"name":"Volume","rating":"good" if vr>1.2 else "neutral" if vr>0.8 else "bad","detail":f"{vr:.2f}x"})
    s = raw["session"]
    f.append({"name":"Session","rating":"good" if s in("london","newyork") else "neutral" if s=="asia" else "bad",
               "detail":s.capitalize()})
    return f

# ── Stats ─────────────────────────────────────────────────────────────────────
def calc_stats(trades):
    closed = [t for t in trades if t.get("result") in ("win","loss")]
    if not closed:
        return {"total":0,"wins":0,"losses":0,"win_rate":0,"total_pnl_usd":0.0,
                "avg_win_pts":0.0,"avg_loss_pts":0.0,"best_trade":0.0,"worst_trade":0.0,
                "by_session":{},"by_score":{},"by_trade_type":{}}
    wins    = [t for t in closed if t["result"]=="win"]
    losses  = [t for t in closed if t["result"]=="loss"]
    pnl_pts = [t.get("pnl_pts",0) or 0 for t in closed]

    by_session = {}
    for t in closed:
        s = t.get("session","unknown")
        if s not in by_session: by_session[s] = {"wins":0,"total":0,"pnl":0.0}
        by_session[s]["total"] += 1
        by_session[s]["pnl"]   += (t.get("pnl_pts",0) or 0) * MNQ_PTS_TO_USD
        if t["result"]=="win": by_session[s]["wins"] += 1
    for s in by_session:
        d = by_session[s]; d["wr"] = round(d["wins"]/d["total"]*100,1) if d["total"] else 0

    by_score = {}
    for t in closed:
        sc = t.get("score",0) or 0
        b  = f"{int(sc//10)*10}-{int(sc//10)*10+10}"
        if b not in by_score: by_score[b] = {"wins":0,"total":0}
        by_score[b]["total"] += 1
        if t["result"]=="win": by_score[b]["wins"] += 1
    for b in by_score:
        d = by_score[b]; d["wr"] = round(d["wins"]/d["total"]*100,1) if d["total"] else 0

    # By trade type
    by_trade_type = {}
    for t in closed:
        tt = t.get("trade_type","unknown")
        if tt not in by_trade_type: by_trade_type[tt] = {"wins":0,"total":0}
        by_trade_type[tt]["total"] += 1
        if t["result"]=="win": by_trade_type[tt]["wins"] += 1
    for tt in by_trade_type:
        d = by_trade_type[tt]; d["wr"] = round(d["wins"]/d["total"]*100,1) if d["total"] else 0

    win_pts  = [t.get("pnl_pts",0) or 0 for t in wins]
    loss_pts = [t.get("pnl_pts",0) or 0 for t in losses]
    return {
        "total": len(closed), "wins": len(wins), "losses": len(losses),
        "win_rate":     round(len(wins)/len(closed)*100,1),
        "total_pnl_usd":round(sum(pnl_pts)*MNQ_PTS_TO_USD,2),
        "avg_win_pts":  round(sum(win_pts)/len(win_pts),1) if win_pts else 0,
        "avg_loss_pts": round(sum(loss_pts)/len(loss_pts),1) if loss_pts else 0,
        "best_trade":   round(max(pnl_pts)*MNQ_PTS_TO_USD,2) if pnl_pts else 0,
        "worst_trade":  round(min(pnl_pts)*MNQ_PTS_TO_USD,2) if pnl_pts else 0,
        "by_session": by_session, "by_score": by_score,
        "by_trade_type": by_trade_type,
    }

# ── Auto AI Retraining ────────────────────────────────────────────────────────
def retrain_model_async(closed_trade):
    def _retrain():
        global model, state
        with retrain_lock:
            try:
                trades = load_trades()
                closed = [t for t in trades if t.get("result") in ("win","loss")]
                if len(closed) < 5:
                    print(f"⏭️  Skipping retrain — need ≥5 closed trades (have {len(closed)})")
                    return
                print(f"🧠 Retraining on {len(closed)} trades...")
                rows = []
                for t in closed:
                    rows.append({
                        "timeframe": "1H", "rsi_at_entry": t.get("rsi_at_entry", 50.0),
                        "ema_diff": t.get("ema_diff", 0.0), "volume_ratio": t.get("volume_ratio", 1.0),
                        "session": t.get("session", "newyork"), "htf_bias": t.get("htf_bias", "bullish"),
                        "trade_direction": t.get("direction", "long"),
                        "sl_distance_points": abs(t.get("sl",0) - t.get("entry",0)),
                        "entry_price": t.get("entry", 0),
                        "result": 1 if t["result"]=="win" else 0,
                    })
                df = pd.DataFrame(rows)
                new_les = {}
                for col in ["timeframe","session","htf_bias","trade_direction"]:
                    le = LabelEncoder()
                    old_classes = list(les[col].classes_) if col in les else []
                    all_vals = list(set(old_classes + df[col].astype(str).tolist()))
                    le.fit(all_vals); new_les[col] = le
                new_les["rsi_zone"] = les["rsi_zone"]
                new_les["vol_tier"] = les["vol_tier"]
                X_rows = []; y = []
                for _, row in df.iterrows():
                    raw = row.to_dict()
                    try:
                        orig_les = les.copy(); les.update(new_les)
                        feat = engineer({
                            "timeframe": raw["timeframe"], "rsi_at_entry": raw["rsi_at_entry"],
                            "ema_diff": raw["ema_diff"], "volume_ratio": raw["volume_ratio"],
                            "session": raw["session"], "htf_bias": raw["htf_bias"],
                            "trade_direction": raw["trade_direction"],
                            "sl_distance_points": raw["sl_distance_points"],
                            "entry_price": raw["entry_price"],
                        })
                        les.update(orig_les)
                        X_rows.append(feat.values[0]); y.append(raw["result"])
                    except Exception as e:
                        print(f"  Skipping row: {e}")
                if len(X_rows) < 5: return
                X = np.array(X_rows); y = np.array(y)
                new_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                    min_samples_leaf=2, random_state=42, class_weight="balanced")
                new_model.fit(X, y)
                joblib.dump({"model": new_model, "label_encoders": les, "feature_cols": FCOLS}, "model.pkl")
                model = new_model
                state["model_version"] += 1
                state["last_retrain"]   = datetime.now().strftime("%H:%M:%S")
                print(f"✅ Retrain complete — v{state['model_version']} | {len(X)} samples")
            except Exception as e:
                import traceback; print(f"❌ Retrain failed: {e}\n{traceback.format_exc()}")
    threading.Thread(target=_retrain, daemon=True).start()

# ── Close trade ───────────────────────────────────────────────────────────────
def _close_active(result, at, price):
    at["result"]     = result
    at["exit_time"]  = datetime.now().strftime("%H:%M:%S")
    at["exit_price"] = round(price, 2)
    at["pnl_pts"]    = round(abs(at["tp"]-at["entry"]),1) if result=="win" else round(-abs(at["sl"]-at["entry"]),1)
    at["pnl_usd"]    = round(at["pnl_pts"] * MNQ_PTS_TO_USD, 2)
    trades = load_trades(); trades.append(at); save_trades(trades)
    state["active_trade"] = None
    print(f"🏁 {result.upper()} {at['pnl_pts']:+.1f}pts (${at['pnl_usd']:+.2f}) [{at.get('trade_type','—')}]")
    retrain_model_async(at)

# ── Main loop ─────────────────────────────────────────────────────────────────
def fetch_and_score():
    global state, prev_score, last_fetch
    now = time.time()
    if now - last_fetch < INTERVAL: return
    last_fetch = now

    try:
        # Fetch all timeframes
        all_bars = fetch_all_timeframes()
        bars_5m  = all_bars.get("5m")

        if bars_5m is None or len(bars_5m) < 10:
            state["error"] = "No bar data — market may be closed"; return

        # Compute indicators on 5m
        bars_5m["ema9"]      = calc_ema(bars_5m["Close"], 9)
        bars_5m["ema21"]     = calc_ema(bars_5m["Close"], 21)
        bars_5m["rsi"]       = calc_rsi(bars_5m["Close"], 14)
        bars_5m["vol_ma"]    = bars_5m["Volume"].rolling(20).mean()
        bars_5m["atr"]       = calc_atr(bars_5m["High"], bars_5m["Low"], bars_5m["Close"], 14)
        bars_5m["ema_diff"]  = bars_5m["ema9"] - bars_5m["ema21"]
        bars_5m["vol_ratio"] = bars_5m["Volume"] / bars_5m["vol_ma"].replace(0, np.nan)

        last      = bars_5m.iloc[-1]
        price     = float(last["Close"])
        cur_high  = float(last["High"])
        cur_low   = float(last["Low"])
        rsi_val   = float(last["rsi"])
        ema_diff  = float(last["ema_diff"])
        vol_ratio = float(last["vol_ratio"]) if not np.isnan(last["vol_ratio"]) else 1.0
        atr_val   = float(last["atr"])       if not np.isnan(last["atr"])       else 10.0
        session   = get_kill_zone()[0]
        htf_bias  = "bullish" if ema_diff > 0 else "bearish"
        sl_dist   = round(atr_val * 1.5, 1)

        # Monitor active trade
        at = state.get("active_trade")
        if at and at["result"] == "open":
            if at["direction"] == "long":
                if cur_high >= at["tp"]:   _close_active("win",  at, price)
                elif cur_low <= at["sl"]:  _close_active("loss", at, price)
            else:
                if cur_low  <= at["tp"]:   _close_active("win",  at, price)
                elif cur_high >= at["sl"]: _close_active("loss", at, price)

        # XGBoost base score (both directions)
        def mk_raw(d):
            return {"timeframe":"1H", "rsi_at_entry":round(rsi_val,1),
                    "ema_diff":round(ema_diff,1), "volume_ratio":round(vol_ratio,2),
                    "session":session, "htf_bias":htf_bias, "trade_direction":d,
                    "sl_distance_points":sl_dist, "entry_price":round(price,2)}

        prob_long  = float(model.predict_proba(engineer(mk_raw("long")))[0,1])
        prob_short = float(model.predict_proba(engineer(mk_raw("short")))[0,1])
        sl_long    = int(round(prob_long  * 100))
        ss_short   = int(round(prob_short * 100))

        if sl_long >= ss_short:
            direction="long";  base_score=sl_long;  prob=prob_long;  raw=mk_raw("long")
            sl_price=round(price-sl_dist,2);   tp_price=round(price+sl_dist*RR,2)
        else:
            direction="short"; base_score=ss_short; prob=prob_short; raw=mk_raw("short")
            sl_price=round(price+sl_dist,2);   tp_price=round(price-sl_dist*RR,2)

        # ICT layer
        ict_score, ict_factors, trade_type, mtf = run_ict_analysis(all_bars, direction, price)

        # Combined final score
        final_score = max(0, min(100, base_score + ict_score))

        take  = final_score >= SCORE_THRESHOLD
        alert = take and prev_score < SCORE_THRESHOLD
        alert_msg = ""

        if alert and not state.get("active_trade"):
            new_trade = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction, "entry": price,
                "sl": sl_price, "tp": tp_price,
                "score": final_score, "base_score": base_score, "ict_score": ict_score,
                "session": session, "htf_bias": htf_bias,
                "rsi_at_entry": round(rsi_val,1), "ema_diff": round(ema_diff,1),
                "volume_ratio": round(vol_ratio,2),
                "trade_type": trade_type,
                "result": "open", "exit_price": None,
                "exit_time": None, "pnl_pts": None, "pnl_usd": None,
            }
            state["active_trade"] = new_trade
            alert_msg = f"AUTO [{trade_type.upper()}]: {direction.upper()} @ {price:,.1f} | Score:{final_score}% (Base:{base_score}+ICT:{ict_score:+d})"
            print(f"🤖 {alert_msg}")

        prev_score = final_score
        trades = load_trades()
        stats  = calc_stats(trades)

        state.update({
            "score": final_score, "base_score": base_score, "ict_score": ict_score,
            "prob": round(prob,3), "take": take, "direction": direction.upper(),
            "score_long": sl_long, "score_short": ss_short,
            "rsi": round(rsi_val,1), "ema_diff": round(ema_diff,1),
            "vol_ratio": round(vol_ratio,2), "session": session,
            "htf_bias": htf_bias, "sl_dist": sl_dist,
            "price": round(price,2), "sl_price": sl_price, "tp_price": tp_price,
            "last_update": datetime.now().strftime("%H:%M:%S"),
            "error": None, "factors": get_base_factors(raw),
            "ict_factors": ict_factors, "trade_type": trade_type,
            "alert": alert, "alert_msg": alert_msg,
            "total_trades": stats["total"], "wins": stats["wins"],
            "losses": stats["losses"], "win_rate": stats["win_rate"],
            "total_pnl_usd": stats["total_pnl_usd"],
            "trades": trades, "stats": stats, "mtf": mtf,
        })

        print(f"[v7] {direction.upper()} base={base_score} ict={ict_score:+d} final={final_score} type={trade_type} price={price}")

    except Exception as e:
        import traceback
        state["error"] = str(e)
        state["last_update"] = datetime.now().strftime("%H:%M:%S")
        print(f"Error: {e}\n{traceback.format_exc()}")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>IFVG Live AI v7</title>
<style>
:root{--bg:#f8f9fa;--surface:#ffffff;--border:#e2e8f0;--text:#1a202c;--muted:#718096;--green:#00a67e;--red:#e53e3e;--amber:#d97706;--blue:#2b6cb0;--purple:#6b46c1}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,sans-serif;padding:16px}
.wrap{max-width:520px;margin:0 auto}
.logo{font-size:10px;color:var(--muted);letter-spacing:3px;margin-bottom:6px;font-family:'Courier New',monospace}
h1{font-size:22px;font-weight:800;margin-bottom:4px}
h1 span{color:var(--green)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.meta-row{display:flex;gap:6px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.meta-badge{font-size:10px;padding:3px 8px;border-radius:4px;font-family:'Courier New',monospace}
.badge-src{background:rgba(77,159,255,.15);color:var(--blue);border:1px solid rgba(77,159,255,.3)}
.badge-model{background:rgba(0,229,160,.1);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.badge-type{background:rgba(176,111,255,.15);color:var(--purple);border:1px solid rgba(176,111,255,.3)}
.badge-retrain{background:rgba(255,182,39,.1);color:var(--amber);border:1px solid rgba(255,182,39,.3)}
.verdict{border-radius:14px;padding:20px;text-align:center;margin-bottom:14px;transition:all 0.4s}
.verdict.take{background:rgba(0,229,160,.08);border:2px solid rgba(0,229,160,.4)}
.verdict.skip{background:rgba(255,77,109,.08);border:2px solid rgba(255,77,109,.3)}
.verdict.loading{background:rgba(255,255,255,.03);border:2px solid var(--border)}
.v-label{font-size:10px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:6px}
.v-text{font-size:34px;font-weight:800}
.verdict.take .v-text{color:var(--green)}.verdict.skip .v-text{color:var(--red)}.verdict.loading .v-text{color:var(--muted)}
.v-dir{font-size:11px;color:var(--muted);margin-top:3px;font-family:'Courier New',monospace}
.score-breakdown{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:10px 0 0}
.score-box{background:var(--bg);border-radius:6px;padding:7px;text-align:center}
.score-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:2px}
.score-val{font-size:16px;font-weight:800}
.bar-wrap{margin:10px 0 2px}
.bar-label{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:5px}
.bar-track{height:6px;background:var(--border);border-radius:99px;overflow:hidden}
.bar-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1),background .4s}
.dual{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.dual-card{border-radius:12px;padding:12px;text-align:center;background:var(--surface);border:1px solid var(--border);transition:border .3s}
.dual-lbl{font-size:9px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:5px}
.dual-val{font-size:22px;font-weight:800}
.dual-sub{font-size:10px;color:var(--muted);margin-top:2px}
.sltp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.sl-box{border-radius:10px;padding:12px;text-align:center;background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3)}
.tp-box{border-radius:10px;padding:12px;text-align:center;background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3)}
.sltp-lbl{font-size:9px;font-family:'Courier New',monospace;letter-spacing:2px;margin-bottom:3px}
.sl-box .sltp-lbl{color:var(--red)}.tp-box .sltp-lbl{color:var(--green)}
.sltp-price{font-size:17px;font-weight:800}
.sl-box .sltp-price{color:var(--red)}.tp-box .sltp-price{color:var(--green)}
.sltp-dist{font-size:10px;color:var(--muted);margin-top:2px}
.mtf-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:12px}
.mtf-cell{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 4px;text-align:center}
.mtf-tf{font-size:8px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:2px}
.mtf-val{font-size:10px;font-weight:700}
.mtf-bull{color:var(--green)}.mtf-bear{color:var(--red)}.mtf-range{color:var(--amber)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:10px}
.card-title{font-size:9px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:10px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.metric{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px 11px}
.m-lbl{font-size:9px;color:var(--muted);letter-spacing:1px;font-family:'Courier New',monospace;margin-bottom:3px}
.m-val{font-size:14px;font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}.purple{color:var(--purple)}
.factors{display:flex;flex-direction:column;gap:6px}
.factor{display:flex;justify-content:space-between;align-items:center;padding:8px 11px;border-radius:8px;background:var(--bg);border:1px solid var(--border)}
.f-left{display:flex;align-items:center;gap:8px}
.f-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.f-name{font-size:11px;color:var(--muted);font-family:'Courier New',monospace}
.f-val{font-size:11px}
.good .f-dot{background:var(--green)}.good .f-val{color:var(--green)}
.neutral .f-dot{background:var(--amber)}.neutral .f-val{color:var(--amber)}
.bad .f-dot{background:var(--red)}.bad .f-val{color:var(--red)}
.ict-factor{display:flex;justify-content:space-between;align-items:center;padding:7px 11px;border-radius:7px;background:var(--bg);border:1px solid var(--border);margin-bottom:5px}
.ict-left{display:flex;align-items:center;gap:7px}
.ict-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.ict-aligned .ict-dot{background:var(--green)}.ict-misaligned .ict-dot{background:var(--red)}
.ict-name{font-size:10px;color:var(--muted);font-family:'Courier New',monospace}
.ict-val{font-size:10px;font-weight:700}
.ict-aligned .ict-val{color:var(--green)}.ict-misaligned .ict-val{color:var(--red)}
.ict-detail{font-size:9px;color:var(--muted);margin-top:2px;font-family:'Courier New',monospace}
.at-card{border-radius:14px;padding:18px;margin-bottom:12px;background:rgba(255,182,39,.08);border:2px solid rgba(255,182,39,.4);display:none}
.at-title{font-size:10px;color:var(--amber);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:10px}
.at-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.at-cell{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px;text-align:center}
.at-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:3px}
.at-val{font-size:16px;font-weight:800}
.at-pnl{margin-top:8px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px}
.at-btns{display:flex;gap:8px;margin-top:8px}
.btn-win{flex:1;padding:9px;border-radius:8px;border:none;background:rgba(0,229,160,.2);color:var(--green);font-weight:700;cursor:pointer;font-size:12px}
.btn-loss{flex:1;padding:9px;border-radius:8px;border:none;background:rgba(255,77,109,.2);color:var(--red);font-weight:700;cursor:pointer;font-size:12px}
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:10px}
.stat{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px;text-align:center}
.stat-val{font-size:18px;font-weight:800}
.stat-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-top:2px}
.pnl-big{text-align:center;padding:12px;border-radius:10px;margin-bottom:10px}
.pnl-big.pos{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.3)}
.pnl-big.neg{background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.3)}
.pnl-label{font-size:10px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:3px}
.pnl-val{font-size:26px;font-weight:800}
.breakdown-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--border);font-size:11px}
.breakdown-row:last-child{border-bottom:none}
.breakdown-label{color:var(--muted);font-family:'Courier New',monospace}
.mini-bar{height:3px;border-radius:2px;background:var(--border);margin-top:3px;overflow:hidden}
.mini-bar-fill{height:100%;border-radius:2px;background:var(--green)}
.tlog-wrap{overflow-x:auto}
.tlog{width:100%;border-collapse:collapse;font-size:10px;font-family:'Courier New',monospace}
.tlog th{font-size:8px;color:var(--muted);letter-spacing:1px;padding:5px 6px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
.tlog td{padding:7px 6px;border-bottom:1px solid rgba(30,34,41,.6);white-space:nowrap}
.tlog tr:hover td{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9px;font-weight:700}
.badge-win{background:rgba(0,229,160,.15);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.badge-loss{background:rgba(255,77,109,.15);color:var(--red);border:1px solid rgba(255,77,109,.3)}
.badge-open{background:rgba(255,182,39,.15);color:var(--amber);border:1px solid rgba(255,182,39,.3)}
.badge-cont{background:rgba(0,229,160,.1);color:var(--green);border:1px solid rgba(0,229,160,.2);font-size:8px}
.badge-rev{background:rgba(176,111,255,.1);color:var(--purple);border:1px solid rgba(176,111,255,.2);font-size:8px}
.dir-long{color:var(--green)}.dir-short{color:var(--red)}
.empty-log{color:var(--muted);font-size:11px;padding:14px 0;text-align:center}
.alert-banner{background:rgba(0,229,160,.15);border:1px solid var(--green);border-radius:10px;padding:12px;margin-bottom:12px;font-size:12px;font-weight:700;color:var(--green);text-align:center;display:none}
.error-box{background:rgba(255,77,109,.1);border:1px solid var(--red);border-radius:10px;padding:10px;font-size:11px;color:var(--red);margin-bottom:10px;display:none}
.update-row{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:'Courier New',monospace;margin-top:4px}
.tabs{display:flex;gap:2px;margin-bottom:12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:3px}
.tab-btn{flex:1;padding:7px;border-radius:7px;border:none;background:transparent;color:var(--muted);font-size:10px;font-family:'Courier New',monospace;letter-spacing:1px;cursor:pointer;transition:all .2s}
.tab-btn.active{background:var(--bg);color:var(--text);font-weight:700}
.tab-pane{display:none}.tab-pane.active{display:block}
</style></head>
<body><div class="wrap">

<div class="logo">NQ / MNQ FUTURES — ICT FRAMEWORK</div>
<h1><span class="dot"></span>IFVG <span>Live AI v7</span></h1>
<div class="meta-row">
  <span class="meta-badge badge-src" id="src-badge">MTF ACTIVE</span>
  <span class="meta-badge badge-model" id="model-badge">MODEL v0</span>
  <span class="meta-badge badge-type" id="type-badge">—</span>
  <span class="meta-badge badge-retrain" id="retrain-badge">AWAITING TRADES</span>
</div>

<div class="alert-banner" id="ab"></div>
<div class="error-box" id="eb"></div>

<!-- Verdict -->
<div class="verdict loading" id="verdict">
  <div class="v-label">AI RECOMMENDATION</div>
  <div class="v-text" id="vt">LOADING...</div>
  <div class="v-dir" id="vd">Fetching MTF data...</div>
  <div class="score-breakdown">
    <div class="score-box"><div class="score-lbl">BASE</div><div class="score-val blue" id="base-score">—</div></div>
    <div class="score-box"><div class="score-lbl">ICT</div><div class="score-val purple" id="ict-score">—</div></div>
    <div class="score-box"><div class="score-lbl">FINAL</div><div class="score-val" id="final-score">—</div></div>
  </div>
  <div class="bar-wrap">
    <div class="bar-label"><span>WIN PROBABILITY</span><span id="bp">—</span></div>
    <div class="bar-track"><div class="bar-fill" id="bf" style="width:0%;background:var(--muted)"></div></div>
  </div>
</div>

<!-- Long/Short -->
<div class="dual">
  <div class="dual-card" id="lcard"><div class="dual-lbl">▲ LONG</div><div class="dual-val" id="lscore">—</div><div class="dual-sub">base score</div></div>
  <div class="dual-card" id="scard"><div class="dual-lbl">▼ SHORT</div><div class="dual-val" id="sscore">—</div><div class="dual-sub">base score</div></div>
</div>

<!-- SL/TP -->
<div class="sltp">
  <div class="sl-box"><div class="sltp-lbl">STOP LOSS</div><div class="sltp-price" id="sl-p">—</div><div class="sltp-dist" id="sl-d">—</div></div>
  <div class="tp-box"><div class="sltp-lbl">TAKE PROFIT</div><div class="sltp-price" id="tp-p">—</div><div class="sltp-dist" id="tp-d">—</div></div>
</div>

<!-- MTF Structure -->
<div class="mtf-grid" id="mtf-grid">
  <div class="mtf-cell"><div class="mtf-tf">WEEKLY</div><div class="mtf-val" id="mtf-w">—</div></div>
  <div class="mtf-cell"><div class="mtf-tf">DAILY</div><div class="mtf-val" id="mtf-d">—</div></div>
  <div class="mtf-cell"><div class="mtf-tf">4H</div><div class="mtf-val" id="mtf-4h">—</div></div>
  <div class="mtf-cell"><div class="mtf-tf">1H</div><div class="mtf-val" id="mtf-1h">—</div></div>
  <div class="mtf-cell"><div class="mtf-tf">15M</div><div class="mtf-val" id="mtf-15m">—</div></div>
</div>

<!-- Active Trade -->
<div class="at-card" id="atcard">
  <div class="at-title">🤖 AUTO PAPER TRADE</div>
  <div class="at-grid">
    <div class="at-cell"><div class="at-lbl">DIRECTION</div><div class="at-val" id="at-dir">—</div></div>
    <div class="at-cell"><div class="at-lbl">ENTRY</div><div class="at-val blue" id="at-entry">—</div></div>
    <div class="at-cell" style="background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3)">
      <div class="at-lbl" style="color:var(--red)">SL</div><div class="at-val red" id="at-sl">—</div>
    </div>
    <div class="at-cell" style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3)">
      <div class="at-lbl" style="color:var(--green)">TP</div><div class="at-val green" id="at-tp">—</div>
    </div>
  </div>
  <div class="at-pnl"><div class="at-lbl">LIVE P&L (paper · 1 MNQ)</div><div class="at-val" id="at-pnl">—</div></div>
  <div class="at-btns">
    <button class="btn-win" onclick="closeTrade('win')">✅ Force WIN</button>
    <button class="btn-loss" onclick="closeTrade('loss')">❌ Force LOSS</button>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('metrics',this)">METRICS</button>
  <button class="tab-btn" onclick="showTab('ict',this)">ICT ANALYSIS</button>
  <button class="tab-btn" onclick="showTab('stats',this)">STATS</button>
  <button class="tab-btn" onclick="showTab('log',this)">LOG</button>
</div>

<!-- METRICS TAB -->
<div id="tab-metrics" class="tab-pane active">
  <div class="card">
    <div class="card-title">LIVE METRICS (5M)</div>
    <div class="metrics">
      <div class="metric"><div class="m-lbl">PRICE</div><div class="m-val blue" id="mp">—</div></div>
      <div class="metric"><div class="m-lbl">RSI</div><div class="m-val" id="mr">—</div></div>
      <div class="metric"><div class="m-lbl">EMA DIFF</div><div class="m-val" id="me">—</div></div>
      <div class="metric"><div class="m-lbl">VOLUME</div><div class="m-val" id="mv">—</div></div>
      <div class="metric"><div class="m-lbl">KILL ZONE</div><div class="m-val" id="ms">—</div></div>
      <div class="metric"><div class="m-lbl">HTF BIAS</div><div class="m-val" id="mb">—</div></div>
    </div>
  </div>
  <div class="card"><div class="card-title">MODEL FACTORS</div><div class="factors" id="factors"></div></div>
</div>

<!-- ICT TAB -->
<div id="tab-ict" class="tab-pane">
  <div class="card">
    <div class="card-title">ICT ANALYSIS</div>
    <div id="ict-factors-list"></div>
  </div>
  <div class="card">
    <div class="card-title">MTF CONTEXT</div>
    <div id="mtf-detail"></div>
  </div>
</div>

<!-- STATS TAB -->
<div id="tab-stats" class="tab-pane">
  <div class="card">
    <div class="card-title">PAPER PERFORMANCE</div>
    <div class="pnl-big" id="pnl-big">
      <div class="pnl-label">TOTAL P&L (1 MNQ)</div>
      <div class="pnl-val" id="pnl-val">$0.00</div>
    </div>
    <div class="stats-grid">
      <div class="stat"><div class="stat-val green" id="sw">0</div><div class="stat-lbl">WINS</div></div>
      <div class="stat"><div class="stat-val red" id="sl2">0</div><div class="stat-lbl">LOSSES</div></div>
      <div class="stat"><div class="stat-val" id="sr">—</div><div class="stat-lbl">WIN RATE</div></div>
      <div class="stat"><div class="stat-val amber" id="sa-win">—</div><div class="stat-lbl">AVG WIN</div></div>
      <div class="stat"><div class="stat-val red" id="sa-loss">—</div><div class="stat-lbl">AVG LOSS</div></div>
      <div class="stat"><div class="stat-val blue" id="st-total">0</div><div class="stat-lbl">TOTAL</div></div>
    </div>
  </div>
  <div class="card"><div class="card-title">BY TRADE TYPE</div><div id="type-breakdown"></div></div>
  <div class="card"><div class="card-title">BY SESSION</div><div id="session-breakdown"></div></div>
  <div class="card"><div class="card-title">BY SCORE</div><div id="score-breakdown"></div></div>
</div>

<!-- LOG TAB -->
<div id="tab-log" class="tab-pane">
  <div class="card">
    <div class="card-title">ALL TRADES</div>
    <div class="tlog-wrap">
      <table class="tlog">
        <thead><tr><th>TIME</th><th>TYPE</th><th>DIR</th><th>ENTRY</th><th>SCORE</th><th>SESSION</th><th>EXIT</th><th>PTS</th><th>USD</th><th>RESULT</th></tr></thead>
        <tbody id="tlog-body"></tbody>
      </table>
      <div class="empty-log" id="tlog-empty">No trades logged yet</div>
    </div>
  </div>
</div>

<div class="update-row">
  <span id="ut">Loading...</span>
  <span id="pt" class="blue" style="font-weight:700">NQ1! MTF</span>
</div>

</div>
<script>
function showTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

function structColor(s) {
  if (!s) return 'var(--muted)';
  if (s==='bullish') return 'var(--green)';
  if (s==='bearish') return 'var(--red)';
  return 'var(--amber)';
}

function structClass(s) {
  if (s==='bullish') return 'mtf-bull';
  if (s==='bearish') return 'mtf-bear';
  return 'mtf-range';
}

async function refresh() {
  try {
    const d = await (await fetch('/state')).json();

    const eb=document.getElementById('eb');
    eb.style.display=d.error?'block':'none';
    if(d.error)eb.textContent='⚠️ '+d.error;

    const ab=document.getElementById('ab');
    if(d.alert&&d.alert_msg){ab.style.display='block';ab.textContent='🤖 '+d.alert_msg;playAlert();setTimeout(()=>ab.style.display='none',15000);}

    // Meta
    document.getElementById('model-badge').textContent='MODEL v'+(d.model_version||0);
    document.getElementById('type-badge').textContent=(d.trade_type||'—').toUpperCase().replace('_',' ');
    document.getElementById('retrain-badge').textContent=d.last_retrain!=='—'?'RETRAINED '+d.last_retrain:'AWAITING TRADES';

    // Verdict
    document.getElementById('verdict').className='verdict '+(d.take?'take':'skip');
    document.getElementById('vt').textContent=d.take?'✅ TAKE':'❌ SKIP';
    document.getElementById('vd').textContent=d.direction+' · Final Score '+d.score+'%';
    document.getElementById('base-score').textContent=(d.base_score||0)+'%';
    const ics=d.ict_score||0;
    document.getElementById('ict-score').textContent=(ics>=0?'+':'')+ics;
    const fs=document.getElementById('final-score');
    fs.textContent=d.score+'%'; fs.style.color=d.take?'var(--green)':'var(--red)';
    const bf=document.getElementById('bf');
    document.getElementById('bp').textContent=d.score+'%';
    bf.style.width=d.score+'%'; bf.style.background=d.take?'var(--green)':'var(--red)';

    // Scores
    const ls=d.score_long||0,ss=d.score_short||0;
    const lv=document.getElementById('lscore'),sv=document.getElementById('sscore');
    lv.textContent=ls+'%'; sv.textContent=ss+'%';
    lv.style.color=ls>=70?'var(--green)':ls>=55?'var(--amber)':'var(--red)';
    sv.style.color=ss>=70?'var(--green)':ss>=55?'var(--amber)':'var(--red)';
    document.getElementById('lcard').style.border=ls>=ss?'2px solid var(--green)':'1px solid var(--border)';
    document.getElementById('scard').style.border=ss>ls?'2px solid var(--green)':'1px solid var(--border)';

    // SL/TP
    document.getElementById('sl-p').textContent=d.sl_price?d.sl_price.toLocaleString():'—';
    document.getElementById('tp-p').textContent=d.tp_price?d.tp_price.toLocaleString():'—';
    document.getElementById('sl-d').textContent=d.sl_dist?d.sl_dist+' pts':'—';
    document.getElementById('tp-d').textContent=d.sl_dist?(d.sl_dist*2).toFixed(1)+' pts (2R)':'—';

    // MTF structure bar
    const mtf=d.mtf||{};
    ['w','d','4h','1h','15m'].forEach((k,i)=>{
      const keys=['weekly','daily','4h','1h','15m'];
      const el=document.getElementById('mtf-'+k);
      if(el){
        const s=mtf[keys[i]]||'—';
        el.textContent=s.toUpperCase().slice(0,4);
        el.className='mtf-val '+structClass(s);
      }
    });

    // Metrics
    const rc=v=>v>=45&&v<=65?'green':v<35||v>70?'red':'amber';
    document.getElementById('mp').textContent=d.price?.toLocaleString()||'—';
    const mr=document.getElementById('mr'); mr.className='m-val '+rc(d.rsi); mr.textContent=d.rsi;
    const me=document.getElementById('me'); me.className='m-val '+(d.ema_diff>0?'green':'red');
    me.textContent=(d.ema_diff>0?'+':'')+d.ema_diff;
    const mv=document.getElementById('mv'); mv.className='m-val '+(d.vol_ratio>1.2?'green':d.vol_ratio<0.8?'red':'amber');
    mv.textContent=d.vol_ratio+'x';
    document.getElementById('ms').textContent=(d.session||'—').toUpperCase().replace('_',' ');
    const mb=document.getElementById('mb'); mb.className='m-val '+(d.htf_bias==='bullish'?'green':'red');
    mb.textContent=d.htf_bias;
    document.getElementById('pt').textContent='NQ1! · '+(d.price?.toLocaleString()||'—');

    // Model factors
    document.getElementById('factors').innerHTML=(d.factors||[]).map(f=>
      `<div class="factor ${f.rating}"><div class="f-left"><div class="f-dot"></div><span class="f-name">${f.name}</span></div><span class="f-val">${f.detail}</span></div>`
    ).join('');

    // ICT factors
    document.getElementById('ict-factors-list').innerHTML=(d.ict_factors||[]).map(f=>`
      <div class="ict-factor ${f.aligned?'ict-aligned':'ict-misaligned'}">
        <div class="ict-left"><div class="ict-dot"></div>
          <div><div class="ict-name">${f.name}</div><div class="ict-detail">${f.detail||''}</div></div>
        </div>
        <div class="ict-val">${f.value||'—'}</div>
      </div>`).join('') || '<div class="empty-log">No ICT data yet</div>';

    // MTF detail
    document.getElementById('mtf-detail').innerHTML = mtf.weekly ? `
      <div class="breakdown-row"><span class="breakdown-label">KZ Active</span><span style="color:${mtf.kz_active?'var(--green)':'var(--muted)'};font-weight:700">${(mtf.kz||'—').toUpperCase().replace('_',' ')}</span></div>
      <div class="breakdown-row"><span class="breakdown-label">Daily FVG Target</span><span style="color:var(--blue);font-weight:700">${mtf.d_fvg_target||'None'}</span></div>
      <div class="breakdown-row"><span class="breakdown-label">TF Alignment</span><span style="color:${(mtf.alignment||0)>=4?'var(--green)':(mtf.alignment||0)>=3?'var(--amber)':'var(--red)'};font-weight:700">${mtf.alignment||0}/5</span></div>
      <div class="breakdown-row"><span class="breakdown-label">HTF Agreement</span><span style="color:${mtf.htf_agree?'var(--green)':'var(--red)'};font-weight:700">${mtf.htf_agree?'YES':'NO'}</span></div>
      <div class="breakdown-row"><span class="breakdown-label">Liq Sweep</span><span style="color:${mtf.sweep?.swept?'var(--purple)':'var(--muted)'};font-weight:700">${mtf.sweep?.swept?(mtf.sweep.type||'').toUpperCase().replace('_',' '):'NONE'}</span></div>
      <div class="breakdown-row"><span class="breakdown-label">Absorption</span><span style="color:${mtf.absorption?.absorbed?'var(--green)':'var(--muted)'};font-weight:700">${mtf.absorption?.absorbed?mtf.absorption.type.toUpperCase()+' '+mtf.absorption.vol_ratio+'x':'NONE'}</span></div>
    ` : '<div class="empty-log">Fetching MTF data...</div>';

    // Active trade
    const atc=document.getElementById('atcard');
    if(d.active_trade&&d.active_trade.result==='open'){
      atc.style.display='block';
      const at=d.active_trade;
      const de=document.getElementById('at-dir');
      de.textContent=at.direction?.toUpperCase(); de.style.color=at.direction==='long'?'var(--green)':'var(--red)';
      document.getElementById('at-entry').textContent=at.entry?.toLocaleString()||'—';
      document.getElementById('at-sl').textContent=at.sl?.toLocaleString()||'—';
      document.getElementById('at-tp').textContent=at.tp?.toLocaleString()||'—';
      const pnl=at.direction==='long'?d.price-at.entry:at.entry-d.price;
      const pnlUsd=pnl*2;
      const pe=document.getElementById('at-pnl');
      pe.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+' pts ('+(pnlUsd>=0?'+':'-')+'$'+Math.abs(pnlUsd).toFixed(2)+')';
      pe.style.color=pnl>=0?'var(--green)':'var(--red)';
    } else { atc.style.display='none'; }

    // Stats
    const st=d.stats||{};
    const pnl=st.total_pnl_usd||0;
    const pb=document.getElementById('pnl-big');
    pb.className='pnl-big '+(pnl>=0?'pos':'neg');
    const pv=document.getElementById('pnl-val');
    pv.textContent=(pnl>=0?'+':'-')+' $'+Math.abs(pnl).toFixed(2); pv.style.color=pnl>=0?'var(--green)':'var(--red)';
    document.getElementById('sw').textContent=st.wins||0;
    document.getElementById('sl2').textContent=st.losses||0;
    document.getElementById('st-total').textContent=st.total||0;
    const sr=document.getElementById('sr');
    sr.textContent=st.total?st.win_rate+'%':'—';
    sr.className='stat-val '+(st.win_rate>=70?'green':st.win_rate>=50?'amber':'red');
    document.getElementById('sa-win').textContent=st.avg_win_pts?'+'+st.avg_win_pts+'pts':'—';
    document.getElementById('sa-loss').textContent=st.avg_loss_pts?st.avg_loss_pts+'pts':'—';

    // Trade type breakdown
    const ttb=st.by_trade_type||{};
    document.getElementById('type-breakdown').innerHTML=Object.keys(ttb).length
      ?Object.entries(ttb).sort((a,b)=>b[1].wr-a[1].wr).map(([tt,v])=>
        `<div class="breakdown-row">
          <div><div class="breakdown-label">${tt.toUpperCase().replace('_',' ')}</div>
          <div class="mini-bar"><div class="mini-bar-fill" style="width:${v.wr}%"></div></div></div>
          <div style="text-align:right">
            <div style="font-weight:700;color:${v.wr>=70?'var(--green)':v.wr>=50?'var(--amber)':'var(--red)'}">${v.wr}%</div>
            <div style="font-size:9px;color:var(--muted)">${v.wins}W/${v.total-v.wins}L</div>
          </div></div>`).join('')
      :'<div class="empty-log">No trades yet</div>';

    // Session breakdown
    const sb=st.by_session||{};
    document.getElementById('session-breakdown').innerHTML=Object.keys(sb).length
      ?Object.entries(sb).sort((a,b)=>b[1].wr-a[1].wr).map(([s,v])=>
        `<div class="breakdown-row">
          <div><div class="breakdown-label">${s.toUpperCase()}</div>
          <div class="mini-bar"><div class="mini-bar-fill" style="width:${v.wr}%"></div></div></div>
          <div style="text-align:right">
            <div style="font-weight:700;color:${v.wr>=70?'var(--green)':v.wr>=50?'var(--amber)':'var(--red)'}">${v.wr}%</div>
            <div style="font-size:9px;color:var(--muted)">${v.wins}W/${v.total-v.wins}L · $${v.pnl.toFixed(0)}</div>
          </div></div>`).join('')
      :'<div class="empty-log">No trades yet</div>';

    // Score breakdown
    const scb=st.by_score||{};
    document.getElementById('score-breakdown').innerHTML=Object.keys(scb).length
      ?Object.entries(scb).sort((a,b)=>a[0].localeCompare(b[0])).map(([range,v])=>
        `<div class="breakdown-row">
          <div><div class="breakdown-label">Score ${range}%</div>
          <div class="mini-bar"><div class="mini-bar-fill" style="width:${v.wr}%"></div></div></div>
          <div style="text-align:right">
            <div style="font-weight:700;color:${v.wr>=70?'var(--green)':v.wr>=50?'var(--amber)':'var(--red)'}">${v.wr}%</div>
            <div style="font-size:9px;color:var(--muted)">${v.wins}W/${v.total-v.wins}L</div>
          </div></div>`).join('')
      :'<div class="empty-log">No trades yet</div>';

    // Trade log
    const trades=(d.trades||[]).slice().reverse();
    const tbody=document.getElementById('tlog-body');
    const empty=document.getElementById('tlog-empty');
    if(trades.length){
      empty.style.display='none';
      tbody.innerHTML=trades.map(t=>{
        const res=t.result||'open';
        const pnlPts=t.pnl_pts!=null?t.pnl_pts:'—';
        const pnlUsd=t.pnl_usd!=null?(t.pnl_usd>=0?'+$':'-$')+Math.abs(t.pnl_usd).toFixed(2):'—';
        const pc=((t.pnl_pts||0)>=0)?'var(--green)':'var(--red)';
        const tt=t.trade_type||'';
        const typeBadge=tt.includes('cont')?`<span class="badge badge-cont">CONT</span>`:tt.includes('rev')?`<span class="badge badge-rev">REV</span>`:'—';
        return `<tr>
          <td style="color:var(--muted)">${(t.time||'').slice(11,16)||'—'}</td>
          <td>${typeBadge}</td>
          <td class="dir-${t.direction}">${(t.direction||'').toUpperCase()}</td>
          <td>${t.entry?.toLocaleString()||'—'}</td>
          <td style="color:var(--blue)">${t.score||'—'}%</td>
          <td style="color:var(--muted)">${(t.session||'—').slice(0,6).toUpperCase()}</td>
          <td style="color:var(--muted)">${t.exit_time||'—'}</td>
          <td style="color:${pc}">${pnlPts!=='—'?(pnlPts>=0?'+':'')+pnlPts:'—'}</td>
          <td style="color:${pc}">${pnlUsd}</td>
          <td><span class="badge badge-${res}">${res.toUpperCase()}</span></td>
        </tr>`;
      }).join('');
    } else { empty.style.display='block'; tbody.innerHTML=''; }

    document.getElementById('ut').textContent='Updated '+d.last_update;
  } catch(e){ console.error(e); }
}

function playAlert(){
  try{
    const ctx=new(window.AudioContext||window.webkitAudioContext)();
    [0,150,300].forEach(delay=>{
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=880;o.type='sine';
      g.gain.setValueAtTime(0,ctx.currentTime+delay/1000);
      g.gain.linearRampToValueAtTime(0.3,ctx.currentTime+delay/1000+0.05);
      g.gain.linearRampToValueAtTime(0,ctx.currentTime+delay/1000+0.3);
      o.start(ctx.currentTime+delay/1000);o.stop(ctx.currentTime+delay/1000+0.3);
    });
  }catch(e){}
}

async function closeTrade(r){
  await fetch('/close_trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({result:r})});
  refresh();
}

refresh(); setInterval(refresh,5000);
</script></body></html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    fetch_and_score()
    return render_template_string(HTML)

@app.route("/state")
def get_state():
    fetch_and_score()
    return jsonify(state)

@app.route("/close_trade", methods=["POST"])
def close_trade():
    data = freq.get_json(); result = data.get("result")
    at   = state.get("active_trade")
    if at and result in ("win","loss"):
        _close_active(result, at, state.get("price", 0))
        return jsonify({"ok": True})
    return jsonify({"ok": False})

def background_loop():
    """Background thread: fetch and score every INTERVAL seconds."""
    while True:
        try:
            fetch_and_score()
        except Exception as e:
            print(f"Background loop error: {e}")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    print("\n🤖 IFVG Live AI v7 — Full MTF ICT Framework")
    print("📊 Timeframes: Weekly → Daily → 4H → 1H → 15m → 5m")
    print("🎯 ICT: FVG + OB + Liquidity + Structure + Kill Zone")
    print(f"📱 Open: http://localhost:5001\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
