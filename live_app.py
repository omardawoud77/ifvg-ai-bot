"""
live_app.py — IFVG Live AI v6
Full Auto Paper Trading + TradingView Price Feed + Auto AI Learning

NEW in v6:
  - Price source: TradingView (NQ1! CME) via tvdatafeed — same chart you watch
  - Auto AI retraining: every closed trade retrains model.pkl automatically
  - Bot still auto-opens paper trades when score >= 70
  - Auto SL/TP monitoring, full stats, trade log — all from v5

Setup:
  pip install tvdatafeed scikit-learn
  Set TV_USERNAME and TV_PASSWORD below (or leave empty for limited access)
"""

from flask import Flask, jsonify, render_template_string, request as freq
import joblib, numpy as np, pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys, json, os, time, threading

app = Flask(__name__)

# ── Price feed: public Yahoo Finance endpoint (NQ=F, no credentials needed) ───
import requests as req
USE_TV = False
print("📡 Price source: Yahoo Finance public endpoint (NQ=F)")

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
MNQ_PTS_TO_USD  = 2.0    # 1 NQ point = $2 for 1 MNQ contract
SCORE_THRESHOLD = 70

last_fetch   = 0
INTERVAL     = 15        # seconds between fetches
retrain_lock = threading.Lock()

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "score": 0, "prob": 0.0, "take": False, "direction": "—",
    "score_long": 0, "score_short": 0,
    "rsi": 50.0, "ema_diff": 0.0, "vol_ratio": 1.0,
    "session": "—", "htf_bias": "—", "sl_dist": 0.0,
    "price": 0.0, "sl_price": 0.0, "tp_price": 0.0,
    "last_update": "—", "error": None, "factors": [],
    "alert": False, "alert_msg": "",
    "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
    "total_pnl_usd": 0.0,
    "active_trade": None,
    "trades": [],
    "stats": {},
    "model_version": 0,       # increments after each retrain
    "last_retrain": "—",
    "data_source": "TradingView" if USE_TV else "Yahoo Finance",
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
    d  = s.diff()
    g  = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(com=p-1, adjust=False).mean()
    al = l.ewm(com=p-1, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def get_session():
    now  = datetime.now(timezone.utc)
    mins = now.hour * 60 + now.minute
    if now.hour < 8:  return "asia"
    if mins < 810:    return "london"
    if mins < 1200:   return "newyork"
    return "overnight"

# ── Price fetch ───────────────────────────────────────────────────────────────
def fetch_bars():
    """Fetch last 100 bars of NQ 5-min data via yfinance."""
    try:
        import yfinance as yf
        bars = yf.download("NQ=F", period="1d", interval="5m", progress=False, auto_adjust=True)
        if bars is None or len(bars) < 10:
            return None
        bars.columns = [c[0] if isinstance(c, tuple) else c for c in bars.columns]
        bars = bars[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return bars if len(bars) >= 10 else None
    except Exception as e:
        print(f"Price fetch error: {e}")
        return None

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
        le  = les[col]; val = str(d[col])
        d[col+"_enc"] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([d])[FCOLS]

def get_factors(raw):
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
                "by_session":{},"by_score":{}}
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
        d = by_session[s]
        d["wr"] = round(d["wins"]/d["total"]*100,1) if d["total"] else 0
    by_score = {}
    for t in closed:
        sc = t.get("score",0) or 0
        b  = f"{int(sc//10)*10}-{int(sc//10)*10+10}"
        if b not in by_score: by_score[b] = {"wins":0,"total":0}
        by_score[b]["total"] += 1
        if t["result"]=="win": by_score[b]["wins"] += 1
    for b in by_score:
        d = by_score[b]
        d["wr"] = round(d["wins"]/d["total"]*100,1) if d["total"] else 0
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
    }

# ── AUTO AI RETRAINING ────────────────────────────────────────────────────────
def retrain_model_async(closed_trade):
    """
    Runs in a background thread after every trade closes.
    Converts the closed trade into a training row and retrains model.pkl.
    The model learns from every paper trade result automatically.
    """
    def _retrain():
        global model, state
        with retrain_lock:
            try:
                trades = load_trades()
                closed = [t for t in trades if t.get("result") in ("win","loss")]
                if len(closed) < 5:
                    print(f"⏭️  Skipping retrain — need ≥5 closed trades (have {len(closed)})")
                    return

                print(f"🧠 Retraining model on {len(closed)} paper trades...")

                # Build training dataframe from all closed trades
                rows = []
                for t in closed:
                    rows.append({
                        "timeframe":           "1H",
                        "rsi_at_entry":        t.get("rsi_at_entry", 50.0),
                        "ema_diff":            t.get("ema_diff", 0.0),
                        "volume_ratio":        t.get("volume_ratio", 1.0),
                        "session":             t.get("session", "newyork"),
                        "htf_bias":            t.get("htf_bias", "bullish"),
                        "trade_direction":     t.get("direction", "long"),
                        "sl_distance_points":  abs(t.get("sl",0) - t.get("entry",0)),
                        "entry_price":         t.get("entry", 0),
                        "result":              1 if t["result"]=="win" else 0,
                    })

                df = pd.DataFrame(rows)

                # Rebuild label encoders from current data + existing classes
                new_les = {}
                cat_cols = ["timeframe","session","htf_bias","trade_direction"]
                for col in cat_cols:
                    le = LabelEncoder()
                    # Combine old classes with new data to avoid unseen label errors
                    old_classes = list(les[col].classes_) if col in les else []
                    new_vals    = df[col].astype(str).tolist()
                    all_vals    = list(set(old_classes + new_vals))
                    le.fit(all_vals)
                    new_les[col] = le

                # Use existing les for rsi_zone and vol_tier (derived, stable)
                new_les["rsi_zone"] = les["rsi_zone"]
                new_les["vol_tier"] = les["vol_tier"]

                # Engineer features for all rows
                X_rows = []
                y      = []
                for _, row in df.iterrows():
                    raw = row.to_dict()
                    raw["trade_direction"] = raw.pop("trade_direction", raw.get("direction","long"))
                    try:
                        # Temporarily swap les for engineering
                        orig_les = les.copy()
                        les.update(new_les)
                        feat_row = engineer({
                            "timeframe":          raw["timeframe"],
                            "rsi_at_entry":       raw["rsi_at_entry"],
                            "ema_diff":           raw["ema_diff"],
                            "volume_ratio":       raw["volume_ratio"],
                            "session":            raw["session"],
                            "htf_bias":           raw["htf_bias"],
                            "trade_direction":    raw["trade_direction"],
                            "sl_distance_points": raw["sl_distance_points"],
                            "entry_price":        raw["entry_price"],
                        })
                        les.update(orig_les)
                        X_rows.append(feat_row.values[0])
                        y.append(raw["result"])
                    except Exception as e:
                        print(f"  Skipping row: {e}")

                if len(X_rows) < 5:
                    print("⚠️  Not enough valid rows after engineering — skipping retrain")
                    return

                X = np.array(X_rows)
                y = np.array(y)

                # Retrain RandomForest
                new_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                )
                new_model.fit(X, y)

                # Save updated model
                joblib.dump({
                    "model":          new_model,
                    "label_encoders": les,
                    "feature_cols":   FCOLS,
                }, "model.pkl")

                # Hot-swap model in memory (no restart needed)
                model = new_model
                state["model_version"] += 1
                state["last_retrain"]   = datetime.now().strftime("%H:%M:%S")

                win_rate = round(sum(y)/len(y)*100, 1)
                print(f"✅ Retrain complete — v{state['model_version']} | {len(X)} samples | dataset WR: {win_rate}%")

            except Exception as e:
                import traceback
                print(f"❌ Retrain failed: {e}\n{traceback.format_exc()}")

    threading.Thread(target=_retrain, daemon=True).start()

# ── Close trade + trigger retrain ─────────────────────────────────────────────
def _close_active(result, at, price):
    at["result"]     = result
    at["exit_time"]  = datetime.now().strftime("%H:%M:%S")
    at["exit_price"] = round(price, 2)
    at["pnl_pts"]    = round(abs(at["tp"]-at["entry"]),1) if result=="win" else round(-abs(at["sl"]-at["entry"]),1)
    at["pnl_usd"]    = round(at["pnl_pts"] * MNQ_PTS_TO_USD, 2)

    # Save to log
    trades = load_trades()
    trades.append(at)
    save_trades(trades)
    state["active_trade"] = None

    print(f"🏁 Trade closed: {result.upper()} {at['pnl_pts']:+.1f}pts (${at['pnl_usd']:+.2f})")

    # Trigger AI retrain in background
    retrain_model_async(at)

# ── Main fetch + score loop ───────────────────────────────────────────────────
def fetch_and_score():
    global state, prev_score, last_fetch
    now = time.time()
    if now - last_fetch < INTERVAL:
        return
    last_fetch = now

    try:
        bars = fetch_bars()
        if bars is None:
            state["error"] = "No bar data — check connection"; return

        bars["ema9"]      = calc_ema(bars["Close"], 9)
        bars["ema21"]     = calc_ema(bars["Close"], 21)
        bars["rsi"]       = calc_rsi(bars["Close"], 14)
        bars["vol_ma"]    = bars["Volume"].rolling(20).mean()
        bars["atr"]       = calc_atr(bars["High"], bars["Low"], bars["Close"], 14)
        bars["ema_diff"]  = bars["ema9"] - bars["ema21"]
        bars["vol_ratio"] = bars["Volume"] / bars["vol_ma"].replace(0, np.nan)

        last      = bars.iloc[-1]
        price     = float(last["Close"])
        cur_high  = float(last["High"])
        cur_low   = float(last["Low"])
        rsi_val   = float(last["rsi"])
        ema_diff  = float(last["ema_diff"])
        vol_ratio = float(last["vol_ratio"]) if not np.isnan(last["vol_ratio"]) else 1.0
        atr_val   = float(last["atr"])       if not np.isnan(last["atr"])       else 10.0
        session   = get_session()
        htf_bias  = "bullish" if ema_diff > 0 else "bearish"
        sl_dist   = round(atr_val * 1.5, 1)

        # ── Monitor active trade SL/TP ────────────────────────────────────────
        at = state.get("active_trade")
        if at and at["result"] == "open":
            if at["direction"] == "long":
                if cur_high >= at["tp"]:   _close_active("win",  at, price)
                elif cur_low <= at["sl"]:  _close_active("loss", at, price)
            else:
                if cur_low  <= at["tp"]:   _close_active("win",  at, price)
                elif cur_high >= at["sl"]: _close_active("loss", at, price)

        # ── Score both directions ─────────────────────────────────────────────
        def mk_raw(d):
            return {
                "timeframe":"1H", "rsi_at_entry":round(rsi_val,1),
                "ema_diff":round(ema_diff,1), "volume_ratio":round(vol_ratio,2),
                "session":session, "htf_bias":htf_bias, "trade_direction":d,
                "sl_distance_points":sl_dist, "entry_price":round(price,2),
            }

        prob_long  = float(model.predict_proba(engineer(mk_raw("long")))[0,1])
        prob_short = float(model.predict_proba(engineer(mk_raw("short")))[0,1])
        sl_long    = int(round(prob_long  * 100))
        ss_short   = int(round(prob_short * 100))

        if sl_long >= ss_short:
            direction="long";  score=sl_long;  prob=prob_long;  raw=mk_raw("long")
            sl_price=round(price-sl_dist,2);   tp_price=round(price+sl_dist*RR,2)
        else:
            direction="short"; score=ss_short; prob=prob_short; raw=mk_raw("short")
            sl_price=round(price+sl_dist,2);   tp_price=round(price-sl_dist*RR,2)

        take  = score >= SCORE_THRESHOLD
        alert = take and prev_score < SCORE_THRESHOLD
        alert_msg = ""

        # ── Auto open trade ───────────────────────────────────────────────────
        if alert and not state.get("active_trade"):
            new_trade = {
                "time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "direction":     direction,
                "entry":         price,
                "sl":            sl_price,
                "tp":            tp_price,
                "score":         score,
                "session":       session,
                "htf_bias":      htf_bias,
                "rsi_at_entry":  round(rsi_val, 1),
                "ema_diff":      round(ema_diff, 1),
                "volume_ratio":  round(vol_ratio, 2),
                "result":        "open",
                "exit_price":    None,
                "exit_time":     None,
                "pnl_pts":       None,
                "pnl_usd":       None,
            }
            state["active_trade"] = new_trade
            alert_msg = f"AUTO TRADE: {direction.upper()} @ {price:,.1f} | SL:{sl_price:,.1f} | TP:{tp_price:,.1f} | Score:{score}%"
            print(f"🤖 Auto trade opened: {direction.upper()} @ {price} score={score}% model_v{state['model_version']}")

        prev_score = score
        trades     = load_trades()
        stats      = calc_stats(trades)

        state.update({
            "score":score, "prob":round(prob,3), "take":take,
            "direction":direction.upper(),
            "score_long":sl_long, "score_short":ss_short,
            "rsi":round(rsi_val,1), "ema_diff":round(ema_diff,1),
            "vol_ratio":round(vol_ratio,2), "session":session,
            "htf_bias":htf_bias, "sl_dist":sl_dist,
            "price":round(price,2), "sl_price":sl_price, "tp_price":tp_price,
            "last_update":datetime.now().strftime("%H:%M:%S"),
            "error":None, "factors":get_factors(raw),
            "alert":alert, "alert_msg":alert_msg,
            "total_trades":stats["total"], "wins":stats["wins"],
            "losses":stats["losses"], "win_rate":stats["win_rate"],
            "total_pnl_usd":stats["total_pnl_usd"],
            "trades":trades, "stats":stats,
        })

        src = "TV" if USE_TV else "YF"
        print(f"[{src}] {direction.upper()} score={score}% price={price} model_v{state['model_version']}")

    except Exception as e:
        import traceback
        state["error"]       = str(e)
        state["last_update"] = datetime.now().strftime("%H:%M:%S")
        print(f"Error: {e}\n{traceback.format_exc()}")


# ── HTML (same as v5 + model version badge) ───────────────────────────────────
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>IFVG Live AI</title>
<style>
:root{--bg:#0a0c0f;--surface:#111318;--border:#1e2229;--text:#e8eaf0;--muted:#5a6070;--green:#00e5a0;--red:#ff4d6d;--amber:#ffb627;--blue:#4d9fff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,sans-serif;padding:16px}
.wrap{max-width:480px;margin:0 auto}
.logo{font-size:10px;color:var(--muted);letter-spacing:3px;margin-bottom:6px;font-family:'Courier New',monospace}
h1{font-size:22px;font-weight:800;margin-bottom:4px}
h1 span{color:var(--green)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.meta-row{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.meta-badge{font-size:10px;padding:3px 8px;border-radius:4px;font-family:'Courier New',monospace;letter-spacing:.05em}
.badge-src{background:rgba(77,159,255,.15);color:var(--blue);border:1px solid rgba(77,159,255,.3)}
.badge-model{background:rgba(0,229,160,.1);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.badge-retrain{background:rgba(255,182,39,.1);color:var(--amber);border:1px solid rgba(255,182,39,.3)}
.verdict{border-radius:14px;padding:24px 20px;text-align:center;margin-bottom:14px;transition:all 0.4s}
.verdict.take{background:rgba(0,229,160,.08);border:2px solid rgba(0,229,160,.4)}
.verdict.skip{background:rgba(255,77,109,.08);border:2px solid rgba(255,77,109,.3)}
.verdict.loading{background:rgba(255,255,255,.03);border:2px solid var(--border)}
.v-label{font-size:10px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:8px}
.v-text{font-size:36px;font-weight:800}
.verdict.take .v-text{color:var(--green)}.verdict.skip .v-text{color:var(--red)}.verdict.loading .v-text{color:var(--muted)}
.v-dir{font-size:12px;color:var(--muted);margin-top:4px;font-family:'Courier New',monospace}
.bar-wrap{margin:14px 0 2px}
.bar-label{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:6px}
.bar-track{height:8px;background:var(--border);border-radius:99px;overflow:hidden}
.bar-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1),background .4s}
.dual{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.dual-card{border-radius:12px;padding:14px;text-align:center;background:var(--surface);border:1px solid var(--border);transition:border .3s}
.dual-lbl{font-size:9px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:6px}
.dual-val{font-size:24px;font-weight:800}
.dual-sub{font-size:10px;color:var(--muted);margin-top:2px}
.sltp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.sl-box{border-radius:10px;padding:14px;text-align:center;background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3)}
.tp-box{border-radius:10px;padding:14px;text-align:center;background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3)}
.sltp-lbl{font-size:9px;font-family:'Courier New',monospace;letter-spacing:2px;margin-bottom:4px}
.sl-box .sltp-lbl{color:var(--red)}.tp-box .sltp-lbl{color:var(--green)}
.sltp-price{font-size:18px;font-weight:800}
.sl-box .sltp-price{color:var(--red)}.tp-box .sltp-price{color:var(--green)}
.sltp-dist{font-size:10px;color:var(--muted);margin-top:2px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:12px}
.card-title{font-size:9px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:12px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.metric{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.m-lbl{font-size:9px;color:var(--muted);letter-spacing:1px;font-family:'Courier New',monospace;margin-bottom:3px}
.m-val{font-size:15px;font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}
.factors{display:flex;flex-direction:column;gap:7px}
.factor{display:flex;justify-content:space-between;align-items:center;padding:9px 12px;border-radius:8px;background:var(--bg);border:1px solid var(--border)}
.f-left{display:flex;align-items:center;gap:8px}
.f-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.f-name{font-size:11px;color:var(--muted);font-family:'Courier New',monospace}
.f-val{font-size:11px}
.good .f-dot{background:var(--green)}.good .f-val{color:var(--green)}
.neutral .f-dot{background:var(--amber)}.neutral .f-val{color:var(--amber)}
.bad .f-dot{background:var(--red)}.bad .f-val{color:var(--red)}
.at-card{border-radius:14px;padding:20px;margin-bottom:14px;background:rgba(255,182,39,.08);border:2px solid rgba(255,182,39,.4);display:none}
.at-title{font-size:10px;color:var(--amber);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:10px}
.at-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.at-cell{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.at-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-bottom:4px}
.at-val{font-size:18px;font-weight:800}
.at-pnl{margin-top:10px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px}
.at-btns{display:flex;gap:8px;margin-top:10px}
.btn-win{flex:1;padding:10px;border-radius:8px;border:none;background:rgba(0,229,160,.2);color:var(--green);font-weight:700;cursor:pointer;font-size:13px}
.btn-loss{flex:1;padding:10px;border-radius:8px;border:none;background:rgba(255,77,109,.2);color:var(--red);font-weight:700;cursor:pointer;font-size:13px}
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.stat{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.stat-val{font-size:20px;font-weight:800}
.stat-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-top:2px}
.pnl-big{text-align:center;padding:14px;border-radius:10px;margin-bottom:12px}
.pnl-big.pos{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.3)}
.pnl-big.neg{background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.3)}
.pnl-label{font-size:10px;color:var(--muted);letter-spacing:2px;font-family:'Courier New',monospace;margin-bottom:4px}
.pnl-val{font-size:28px;font-weight:800}
.breakdown-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--border);font-size:12px}
.breakdown-row:last-child{border-bottom:none}
.breakdown-label{color:var(--muted);font-family:'Courier New',monospace}
.breakdown-wr{font-weight:700}
.mini-bar{height:4px;border-radius:2px;background:var(--border);margin-top:3px;overflow:hidden}
.mini-bar-fill{height:100%;border-radius:2px;background:var(--green)}
.tlog-wrap{overflow-x:auto;margin-top:4px}
.tlog{width:100%;border-collapse:collapse;font-size:11px;font-family:'Courier New',monospace}
.tlog th{font-size:9px;color:var(--muted);letter-spacing:1px;padding:6px 8px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
.tlog td{padding:8px 8px;border-bottom:1px solid rgba(30,34,41,.6);white-space:nowrap;vertical-align:middle}
.tlog tr:last-child td{border-bottom:none}
.tlog tr:hover td{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.badge-win{background:rgba(0,229,160,.15);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.badge-loss{background:rgba(255,77,109,.15);color:var(--red);border:1px solid rgba(255,77,109,.3)}
.badge-open{background:rgba(255,182,39,.15);color:var(--amber);border:1px solid rgba(255,182,39,.3)}
.dir-long{color:var(--green)}.dir-short{color:var(--red)}
.empty-log{color:var(--muted);font-size:11px;font-family:'Courier New',monospace;padding:16px 0;text-align:center}
.alert-banner{background:rgba(0,229,160,.15);border:1px solid var(--green);border-radius:10px;padding:14px;margin-bottom:14px;font-size:13px;font-weight:700;color:var(--green);text-align:center;display:none}
.error-box{background:rgba(255,77,109,.1);border:1px solid var(--red);border-radius:10px;padding:12px;font-size:12px;color:var(--red);margin-bottom:12px;display:none}
.update-row{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:'Courier New',monospace;margin-top:4px}
.tabs{display:flex;gap:2px;margin-bottom:14px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:4px}
.tab-btn{flex:1;padding:8px;border-radius:7px;border:none;background:transparent;color:var(--muted);font-size:11px;font-family:'Courier New',monospace;letter-spacing:1px;cursor:pointer;transition:all .2s}
.tab-btn.active{background:var(--bg);color:var(--text);font-weight:700}
.tab-pane{display:none}.tab-pane.active{display:block}
</style></head>
<body><div class="wrap">

<div class="logo">NQ / MNQ FUTURES</div>
<h1><span class="dot"></span>IFVG <span>Live AI</span></h1>
<div class="meta-row">
  <span class="meta-badge badge-src" id="src-badge">SOURCE: —</span>
  <span class="meta-badge badge-model" id="model-badge">MODEL v0</span>
  <span class="meta-badge badge-retrain" id="retrain-badge">NO RETRAIN YET</span>
</div>

<div class="alert-banner" id="ab"></div>
<div class="error-box" id="eb"></div>

<div class="verdict loading" id="verdict">
  <div class="v-label">AI RECOMMENDATION</div>
  <div class="v-text" id="vt">LOADING...</div>
  <div class="v-dir" id="vd">Connecting to price feed...</div>
  <div class="bar-wrap">
    <div class="bar-label"><span>WIN PROBABILITY</span><span id="bp">—</span></div>
    <div class="bar-track"><div class="bar-fill" id="bf" style="width:0%;background:var(--muted)"></div></div>
  </div>
</div>

<div class="dual">
  <div class="dual-card" id="lcard"><div class="dual-lbl">▲ LONG</div><div class="dual-val" id="lscore">—</div><div class="dual-sub">score</div></div>
  <div class="dual-card" id="scard"><div class="dual-lbl">▼ SHORT</div><div class="dual-val" id="sscore">—</div><div class="dual-sub">score</div></div>
</div>

<div class="sltp">
  <div class="sl-box"><div class="sltp-lbl">STOP LOSS</div><div class="sltp-price" id="sl-p">—</div><div class="sltp-dist" id="sl-d">—</div></div>
  <div class="tp-box"><div class="sltp-lbl">TAKE PROFIT</div><div class="sltp-price" id="tp-p">—</div><div class="sltp-dist" id="tp-d">—</div></div>
</div>

<div class="at-card" id="atcard">
  <div class="at-title">🤖 AUTO PAPER TRADE — MONITORING</div>
  <div class="at-grid">
    <div class="at-cell"><div class="at-lbl">DIRECTION</div><div class="at-val" id="at-dir">—</div></div>
    <div class="at-cell"><div class="at-lbl">ENTRY</div><div class="at-val blue" id="at-entry">—</div></div>
    <div class="at-cell" style="background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3)">
      <div class="at-lbl" style="color:var(--red)">STOP LOSS</div><div class="at-val red" id="at-sl">—</div>
    </div>
    <div class="at-cell" style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3)">
      <div class="at-lbl" style="color:var(--green)">TAKE PROFIT</div><div class="at-val green" id="at-tp">—</div>
    </div>
  </div>
  <div class="at-pnl"><div class="at-lbl">LIVE P&L (paper · 1 MNQ)</div><div class="at-val" id="at-pnl">—</div></div>
  <div class="at-btns">
    <button class="btn-win"  onclick="closeTrade('win')">✅ Force WIN</button>
    <button class="btn-loss" onclick="closeTrade('loss')">❌ Force LOSS</button>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('metrics',this)">METRICS</button>
  <button class="tab-btn" onclick="showTab('stats',this)">STATS</button>
  <button class="tab-btn" onclick="showTab('log',this)">TRADE LOG</button>
</div>

<div id="tab-metrics" class="tab-pane active">
  <div class="card">
    <div class="card-title">LIVE METRICS</div>
    <div class="metrics">
      <div class="metric"><div class="m-lbl">PRICE</div><div class="m-val blue" id="mp">—</div></div>
      <div class="metric"><div class="m-lbl">RSI</div><div class="m-val" id="mr">—</div></div>
      <div class="metric"><div class="m-lbl">EMA DIFF</div><div class="m-val" id="me">—</div></div>
      <div class="metric"><div class="m-lbl">VOLUME</div><div class="m-val" id="mv">—</div></div>
      <div class="metric"><div class="m-lbl">SESSION</div><div class="m-val" id="ms">—</div></div>
      <div class="metric"><div class="m-lbl">HTF BIAS</div><div class="m-val" id="mb">—</div></div>
    </div>
  </div>
  <div class="card"><div class="card-title">FACTOR BREAKDOWN</div><div class="factors" id="factors"></div></div>
</div>

<div id="tab-stats" class="tab-pane">
  <div class="card">
    <div class="card-title">PAPER TRADING PERFORMANCE</div>
    <div class="pnl-big" id="pnl-big">
      <div class="pnl-label">TOTAL P&L (1 MNQ CONTRACT)</div>
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
  <div class="card"><div class="card-title">WIN RATE BY SESSION</div><div id="session-breakdown"></div></div>
  <div class="card"><div class="card-title">WIN RATE BY SCORE RANGE</div><div id="score-breakdown"></div></div>
</div>

<div id="tab-log" class="tab-pane">
  <div class="card">
    <div class="card-title">ALL TRADES</div>
    <div class="tlog-wrap">
      <table class="tlog">
        <thead><tr><th>TIME</th><th>DIR</th><th>ENTRY</th><th>SL</th><th>TP</th><th>SCORE</th><th>SESSION</th><th>EXIT</th><th>PTS</th><th>USD</th><th>RESULT</th></tr></thead>
        <tbody id="tlog-body"></tbody>
      </table>
      <div class="empty-log" id="tlog-empty">No trades logged yet</div>
    </div>
  </div>
</div>

<div class="update-row">
  <span id="ut">Loading...</span>
  <span id="pt" class="blue" style="font-weight:700">NQ1!</span>
</div>

</div>
<script>
function showTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

async function refresh() {
  try {
    const d = await (await fetch('/state')).json();

    // Error
    const eb = document.getElementById('eb');
    eb.style.display = d.error ? 'block' : 'none';
    if (d.error) eb.textContent = '⚠️ ' + d.error;

    // Alert
    const ab = document.getElementById('ab');
    if (d.alert && d.alert_msg) {
      ab.style.display = 'block';
      ab.textContent = '🤖 ' + d.alert_msg;
      playAlert();
      setTimeout(() => ab.style.display='none', 15000);
    }

    // Meta badges
    document.getElementById('src-badge').textContent    = 'SOURCE: ' + (d.data_source || '—');
    document.getElementById('model-badge').textContent  = 'MODEL v' + (d.model_version || 0);
    document.getElementById('retrain-badge').textContent = d.last_retrain !== '—'
      ? 'RETRAINED ' + d.last_retrain : 'AWAITING TRADES';

    // Verdict
    document.getElementById('verdict').className = 'verdict ' + (d.take ? 'take' : 'skip');
    document.getElementById('vt').textContent    = d.take ? '✅ TAKE' : '❌ SKIP';
    document.getElementById('vd').textContent    = d.direction + ' · Score ' + d.score + '%';
    const bf = document.getElementById('bf');
    document.getElementById('bp').textContent = d.score + '%';
    bf.style.width      = d.score + '%';
    bf.style.background = d.take ? 'var(--green)' : 'var(--red)';

    // Scores
    const ls=d.score_long||0, ss=d.score_short||0;
    const lv=document.getElementById('lscore'), sv=document.getElementById('sscore');
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

    // Metrics
    const rc=v=>v>=45&&v<=65?'green':v<35||v>70?'red':'amber';
    document.getElementById('mp').textContent=d.price?.toLocaleString()||'—';
    const mr=document.getElementById('mr'); mr.className='m-val '+rc(d.rsi); mr.textContent=d.rsi;
    const me=document.getElementById('me'); me.className='m-val '+(d.ema_diff>0?'green':'red');
    me.textContent=(d.ema_diff>0?'+':'')+d.ema_diff;
    const mv=document.getElementById('mv'); mv.className='m-val '+(d.vol_ratio>1.2?'green':d.vol_ratio<0.8?'red':'amber');
    mv.textContent=d.vol_ratio+'x';
    document.getElementById('ms').textContent=d.session;
    const mb=document.getElementById('mb'); mb.className='m-val '+(d.htf_bias==='bullish'?'green':'red');
    mb.textContent=d.htf_bias;
    document.getElementById('pt').textContent='NQ1! · '+(d.price?.toLocaleString()||'—');

    // Factors
    document.getElementById('factors').innerHTML=(d.factors||[]).map(f=>
      `<div class="factor ${f.rating}"><div class="f-left"><div class="f-dot"></div><span class="f-name">${f.name}</span></div><span class="f-val">${f.detail}</span></div>`
    ).join('');

    // Active trade
    const atc=document.getElementById('atcard');
    if (d.active_trade && d.active_trade.result==='open') {
      atc.style.display='block';
      const at=d.active_trade;
      const de=document.getElementById('at-dir');
      de.textContent=at.direction?.toUpperCase();
      de.style.color=at.direction==='long'?'var(--green)':'var(--red)';
      document.getElementById('at-entry').textContent=at.entry?.toLocaleString()||'—';
      document.getElementById('at-sl').textContent=at.sl?.toLocaleString()||'—';
      document.getElementById('at-tp').textContent=at.tp?.toLocaleString()||'—';
      const pnl=at.direction==='long'?d.price-at.entry:at.entry-d.price;
      const pnlUsd=pnl*2;
      const pe=document.getElementById('at-pnl');
      pe.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+' pts  ('+(pnlUsd>=0?'+':'-')+'$'+Math.abs(pnlUsd).toFixed(2)+')';
      pe.style.color=pnl>=0?'var(--green)':'var(--red)';
    } else { atc.style.display='none'; }

    // Stats
    const st=d.stats||{};
    const pnl=st.total_pnl_usd||0;
    const pb=document.getElementById('pnl-big');
    pb.className='pnl-big '+(pnl>=0?'pos':'neg');
    const pv=document.getElementById('pnl-val');
    pv.textContent=(pnl>=0?'+':'')+' $'+Math.abs(pnl).toFixed(2);
    pv.style.color=pnl>=0?'var(--green)':'var(--red)';
    document.getElementById('sw').textContent=st.wins||0;
    document.getElementById('sl2').textContent=st.losses||0;
    document.getElementById('st-total').textContent=st.total||0;
    const sr=document.getElementById('sr');
    sr.textContent=st.total?st.win_rate+'%':'—';
    sr.className='stat-val '+(st.win_rate>=70?'green':st.win_rate>=50?'amber':'red');
    document.getElementById('sa-win').textContent=st.avg_win_pts?'+'+st.avg_win_pts+'pts':'—';
    document.getElementById('sa-loss').textContent=st.avg_loss_pts?st.avg_loss_pts+'pts':'—';

    // Session breakdown
    const sb=st.by_session||{};
    document.getElementById('session-breakdown').innerHTML=Object.keys(sb).length
      ?Object.entries(sb).sort((a,b)=>b[1].wr-a[1].wr).map(([s,v])=>
        `<div class="breakdown-row">
          <div><div class="breakdown-label">${s.toUpperCase()}</div>
          <div class="mini-bar"><div class="mini-bar-fill" style="width:${v.wr}%"></div></div></div>
          <div style="text-align:right">
            <div class="breakdown-wr" style="color:${v.wr>=70?'var(--green)':v.wr>=50?'var(--amber)':'var(--red)'}">${v.wr}%</div>
            <div style="font-size:10px;color:var(--muted)">${v.wins}W/${v.total-v.wins}L · $${v.pnl.toFixed(0)}</div>
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
            <div class="breakdown-wr" style="color:${v.wr>=70?'var(--green)':v.wr>=50?'var(--amber)':'var(--red)'}">${v.wr}%</div>
            <div style="font-size:10px;color:var(--muted)">${v.wins}W/${v.total-v.wins}L</div>
          </div></div>`).join('')
      :'<div class="empty-log">No trades yet</div>';

    // Trade log table
    const trades=(d.trades||[]).slice().reverse();
    const tbody=document.getElementById('tlog-body');
    const empty=document.getElementById('tlog-empty');
    if (trades.length) {
      empty.style.display='none';
      tbody.innerHTML=trades.map(t=>{
        const res=t.result||'open';
        const pnlPts=t.pnl_pts!=null?t.pnl_pts:'—';
        const pnlUsd=t.pnl_usd!=null?(t.pnl_usd>=0?'+$':'-$')+Math.abs(t.pnl_usd).toFixed(2):'—';
        const pc=((t.pnl_pts||0)>=0)?'var(--green)':'var(--red)';
        return `<tr>
          <td style="color:var(--muted)">${(t.time||'').slice(11,16)||'—'}</td>
          <td class="dir-${t.direction}">${(t.direction||'').toUpperCase()}</td>
          <td>${t.entry?.toLocaleString()||'—'}</td>
          <td style="color:var(--red)">${t.sl?.toLocaleString()||'—'}</td>
          <td style="color:var(--green)">${t.tp?.toLocaleString()||'—'}</td>
          <td style="color:var(--blue)">${t.score||'—'}%</td>
          <td style="color:var(--muted)">${(t.session||'—').toUpperCase()}</td>
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
    data   = freq.get_json()
    result = data.get("result")
    at     = state.get("active_trade")
    if at and result in ("win","loss"):
        _close_active(result, at, state.get("price", 0))
        return jsonify({"ok": True})
    return jsonify({"ok": False})

if __name__ == "__main__":
    print("\n🤖 IFVG Live AI v6 — Auto Paper Trading + AI Learning")
    print(f"📡 Price source: {'TradingView (NQ1! CME)' if USE_TV else 'Yahoo Finance (fallback)'}")
    print("📱 Open: http://localhost:5001\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
