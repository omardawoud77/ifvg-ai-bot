"""
live_app.py — IFVG Live AI v3
Fixed: uses Flask's before_request for data fetching (no threading issues)
"""

from flask import Flask, jsonify, render_template_string, request as freq
import joblib, numpy as np, pandas as pd, requests as req
from datetime import datetime, timezone
import sys, json, os, time

app = Flask(__name__)

try:
    art   = joblib.load("model.pkl")
    model = art["model"]
    les   = art["label_encoders"]
    FCOLS = art["feature_cols"]
    print("Model loaded")
except:
    sys.exit("model.pkl not found")

TRADE_LOG = "live_trades.json"
RR        = 2.0
last_fetch = 0
INTERVAL   = 15  # seconds between fetches

def load_trades():
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG) as f: return json.load(f)
    return []

def save_trades(t):
    with open(TRADE_LOG,"w") as f: json.dump(t,f,indent=2)

state = {
    "score":0,"prob":0.0,"take":False,"direction":"—",
    "score_long":0,"score_short":0,
    "rsi":50.0,"ema_diff":0.0,"vol_ratio":1.0,
    "session":"—","htf_bias":"—","sl_dist":0.0,
    "price":0.0,"sl_price":0.0,"tp_price":0.0,
    "last_update":"—","error":None,"factors":[],
    "alert":False,"alert_msg":"",
    "total_trades":0,"wins":0,"win_rate":0,
    "active_trade":None,"trades":[],
}
prev_score = 0

def calc_ema(s,p): return s.ewm(span=p,adjust=False).mean()
def calc_rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(com=p-1,adjust=False).mean(); al=l.ewm(com=p-1,adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,np.nan))).fillna(50)
def calc_atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def get_session():
    now=datetime.now(timezone.utc); mins=now.hour*60+now.minute
    if now.hour<8: return "asia"
    if mins<810:   return "london"
    if mins<1200:  return "newyork"
    return "overnight"

def engineer(raw):
    d=raw.copy()
    d["bias_aligned"]=int((d["htf_bias"]=="bullish" and d["trade_direction"]=="long") or (d["htf_bias"]=="bearish" and d["trade_direction"]=="short"))
    d["ema_aligned"]=int((d["ema_diff"]>0 and d["trade_direction"]=="long") or (d["ema_diff"]<0 and d["trade_direction"]=="short"))
    sq={"london":2,"newyork":2,"asia":1,"overnight":0}
    d["session_quality"]=sq.get(d["session"],0)
    d["bias_x_session"]=d["bias_aligned"]*d["session_quality"]
    d["rsi_dist_50"]=abs(d["rsi_at_entry"]-50)
    d["ema_abs"]=abs(d["ema_diff"])
    r=d["rsi_at_entry"]
    d["rsi_zone"]="oversold" if r<=35 else "low_neutral" if r<=45 else "mid_neutral" if r<=55 else "high_neutral" if r<=65 else "overbought"
    v=d["volume_ratio"]
    d["vol_tier"]="very_low" if v<=0.8 else "low" if v<=1.0 else "normal" if v<=1.2 else "high" if v<=1.5 else "very_high"
    for col in ["timeframe","session","htf_bias","trade_direction","rsi_zone","vol_tier"]:
        le=les[col]; val=str(d[col])
        d[col+"_enc"]=int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([d])[FCOLS]

def get_factors(raw):
    f=[]
    al=(raw["htf_bias"]=="bullish" and raw["trade_direction"]=="long") or (raw["htf_bias"]=="bearish" and raw["trade_direction"]=="short")
    f.append({"name":"HTF Bias","rating":"good" if al else "bad","detail":f"{'Aligns' if al else 'Conflicts'} with {raw['trade_direction']}"})
    r=raw["rsi_at_entry"]
    f.append({"name":"RSI","rating":"good" if 45<=r<=65 else "neutral" if 35<r<70 else "bad","detail":f"{r:.1f}"})
    ed=raw["ema_diff"]
    eo=(ed>0 and raw["trade_direction"]=="long") or (ed<0 and raw["trade_direction"]=="short")
    f.append({"name":"EMA","rating":"good" if eo else "bad","detail":f"{ed:+.1f}pts"})
    vr=raw["volume_ratio"]
    f.append({"name":"Volume","rating":"good" if vr>1.2 else "neutral" if vr>0.8 else "bad","detail":f"{vr:.2f}x"})
    s=raw["session"]
    f.append({"name":"Session","rating":"good" if s in("london","newyork") else "neutral" if s=="asia" else "bad","detail":s.capitalize()})
    return f

def fetch_and_score():
    global state, prev_score, last_fetch
    now = time.time()
    if now - last_fetch < INTERVAL:
        return
    last_fetch = now

    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval=5m&range=1d"
        r   = req.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        js  = r.json()
        res = js["chart"]["result"][0]
        ts  = res["timestamp"]
        q   = res["indicators"]["quote"][0]
        bars = pd.DataFrame({
            "Close": q["close"], "High": q["high"],
            "Low":   q["low"],   "Open": q["open"], "Volume": q["volume"],
        }, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()

        if len(bars) < 10:
            state["error"] = "Not enough bars"; return

        bars["ema9"]     = calc_ema(bars["Close"],9)
        bars["ema21"]    = calc_ema(bars["Close"],21)
        bars["rsi"]      = calc_rsi(bars["Close"],14)
        bars["vol_ma"]   = bars["Volume"].rolling(20).mean()
        bars["atr"]      = calc_atr(bars["High"],bars["Low"],bars["Close"],14)
        bars["ema_diff"] = bars["ema9"]-bars["ema21"]
        bars["vol_ratio"]= bars["Volume"]/bars["vol_ma"].replace(0,np.nan)

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
        sl_dist   = round(atr_val*1.5, 1)

        # Check active trade outcome
        at = state.get("active_trade")
        if at and at["result"]=="open":
            if at["direction"]=="long":
                if cur_high>=at["tp"]:   _close_active("win", at, price)
                elif cur_low<=at["sl"]:  _close_active("loss", at, price)
            else:
                if cur_low<=at["tp"]:    _close_active("win", at, price)
                elif cur_high>=at["sl"]: _close_active("loss", at, price)

        # Score both directions
        def mk_raw(d): return {
            "timeframe":"1H","rsi_at_entry":round(rsi_val,1),
            "ema_diff":round(ema_diff,1),"volume_ratio":round(vol_ratio,2),
            "session":session,"htf_bias":htf_bias,"trade_direction":d,
            "sl_distance_points":sl_dist,"entry_price":round(price,2),
        }
        prob_long  = float(model.predict_proba(engineer(mk_raw("long")))[0,1])
        prob_short = float(model.predict_proba(engineer(mk_raw("short")))[0,1])
        sl_long    = int(round(prob_long*100))
        ss_short   = int(round(prob_short*100))

        if sl_long >= ss_short:
            direction=  "long";  score=sl_long;  prob=prob_long;  raw=mk_raw("long")
            sl_price=round(price-sl_dist,2); tp_price=round(price+sl_dist*RR,2)
        else:
            direction= "short"; score=ss_short; prob=prob_short; raw=mk_raw("short")
            sl_price=round(price+sl_dist,2); tp_price=round(price-sl_dist*RR,2)

        take  = score >= 70
        alert = take and prev_score < 70
        alert_msg = f"TAKE {direction.upper()}  Entry:{price:,.1f}  SL:{sl_price:,.1f}  TP:{tp_price:,.1f}  Score:{score}%" if alert else ""

        if alert and not state.get("active_trade"):
            state["active_trade"] = {
                "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "direction":direction,"entry":price,
                "sl":sl_price,"tp":tp_price,"score":score,
                "session":session,"result":"open",
                "exit_price":None,"exit_time":None,"pnl_pts":None,
            }

        prev_score = score
        trades = load_trades()
        closed = [t for t in trades if t["result"] in("win","loss")]
        wins   = sum(1 for t in closed if t["result"]=="win")
        wr     = round(wins/len(closed)*100,1) if closed else 0

        state.update({
            "score":score,"prob":round(prob,3),"take":take,
            "direction":direction.upper(),
            "score_long":sl_long,"score_short":ss_short,
            "rsi":round(rsi_val,1),"ema_diff":round(ema_diff,1),
            "vol_ratio":round(vol_ratio,2),"session":session,
            "htf_bias":htf_bias,"sl_dist":sl_dist,
            "price":round(price,2),"sl_price":sl_price,"tp_price":tp_price,
            "last_update":datetime.now().strftime("%H:%M:%S"),
            "error":None,"factors":get_factors(raw),
            "alert":alert,"alert_msg":alert_msg,
            "total_trades":len(closed),"wins":wins,"win_rate":wr,
            "trades":trades,
        })
        print(f"Updated: {direction.upper()} score={score}% price={price}")

    except Exception as e:
        import traceback
        state["error"] = str(e)
        state["last_update"] = datetime.now().strftime("%H:%M:%S")
        print(f"Error: {e}")
        print(traceback.format_exc())

def _close_active(result, at, price):
    at["result"]     = result
    at["exit_time"]  = datetime.now().strftime("%H:%M:%S")
    at["exit_price"] = price
    at["pnl_pts"]    = round(abs(at["tp"]-at["entry"]) if result=="win" else -abs(at["sl"]-at["entry"]),1)
    trades = load_trades(); trades.append(at); save_trades(trades)
    state["active_trade"] = None
    print(f"Trade closed: {result.upper()} {at['pnl_pts']:+.1f}pts")

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>IFVG Live AI</title>
<style>
:root{--bg:#0a0c0f;--surface:#111318;--border:#1e2229;--text:#e8eaf0;--muted:#5a6070;--green:#00e5a0;--red:#ff4d6d;--amber:#ffb627;--blue:#4d9fff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,sans-serif;padding:16px}
.wrap{max-width:440px;margin:0 auto}
.logo{font-size:10px;color:var(--muted);letter-spacing:3px;margin-bottom:6px;font-family:'Courier New',monospace}
h1{font-size:22px;font-weight:800;margin-bottom:16px}
h1 span{color:var(--green)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
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
.stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.stat{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.stat-val{font-size:20px;font-weight:800}
.stat-lbl{font-size:9px;color:var(--muted);font-family:'Courier New',monospace;margin-top:2px}
.tlist{display:flex;flex-direction:column;gap:6px;max-height:200px;overflow-y:auto}
.titem{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:8px;background:var(--bg);border:1px solid var(--border);font-size:11px;font-family:'Courier New',monospace}
.titem.win{border-left:3px solid var(--green)}.titem.loss{border-left:3px solid var(--red)}.titem.open{border-left:3px solid var(--amber)}
.tres.win{color:var(--green)}.tres.loss{color:var(--red)}.tres.open{color:var(--amber)}
.alert-banner{background:rgba(0,229,160,.15);border:1px solid var(--green);border-radius:10px;padding:14px;margin-bottom:14px;font-size:14px;font-weight:700;color:var(--green);text-align:center;display:none}
.error-box{background:rgba(255,77,109,.1);border:1px solid var(--red);border-radius:10px;padding:12px;font-size:12px;color:var(--red);margin-bottom:12px;display:none}
.update-row{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:'Courier New',monospace;margin-top:4px}
</style></head>
<body><div class="wrap">
<div class="logo">NQ / MNQ FUTURES</div>
<h1><span class="dot"></span>IFVG <span>Live AI</span></h1>
<div class="alert-banner" id="ab"></div>
<div class="error-box" id="eb"></div>
<div class="verdict loading" id="verdict">
  <div class="v-label">AI RECOMMENDATION</div>
  <div class="v-text" id="vt">LOADING...</div>
  <div class="v-dir" id="vd">Fetching live NQ data...</div>
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
  <div class="at-title">⏳ ACTIVE TRADE</div>
  <div class="at-grid">
    <div class="at-cell"><div class="at-lbl">DIRECTION</div><div class="at-val" id="at-dir">—</div></div>
    <div class="at-cell"><div class="at-lbl">ENTRY</div><div class="at-val blue" id="at-entry">—</div></div>
    <div class="at-cell" style="background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3)"><div class="at-lbl" style="color:var(--red)">STOP LOSS</div><div class="at-val red" id="at-sl">—</div></div>
    <div class="at-cell" style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3)"><div class="at-lbl" style="color:var(--green)">TAKE PROFIT</div><div class="at-val green" id="at-tp">—</div></div>
  </div>
  <div class="at-pnl"><div class="at-lbl">LIVE P&L</div><div class="at-val" id="at-pnl">—</div></div>
  <div class="at-btns">
    <button class="btn-win"  onclick="closeTrade('win')">✅ Mark WIN</button>
    <button class="btn-loss" onclick="closeTrade('loss')">❌ Mark LOSS</button>
  </div>
</div>
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
<div class="card">
  <div class="card-title">TRACKED PERFORMANCE</div>
  <div class="stats">
    <div class="stat"><div class="stat-val green" id="sw">0</div><div class="stat-lbl">WINS</div></div>
    <div class="stat"><div class="stat-val red"   id="sl">0</div><div class="stat-lbl">LOSSES</div></div>
    <div class="stat"><div class="stat-val"       id="sr">—</div><div class="stat-lbl">WIN RATE</div></div>
  </div>
</div>
<div class="card"><div class="card-title">TRADE LOG</div><div class="tlist" id="tl"><div style="color:var(--muted);font-size:11px;font-family:'Courier New',monospace">Waiting for signals...</div></div></div>
<div class="update-row"><span id="ut">Loading...</span><span id="pt" class="blue" style="font-weight:700">NQ=F</span></div>
</div>
<script>
async function refresh(){
  try{
    const d=await(await fetch('/state')).json();
    const eb=document.getElementById('eb');
    eb.style.display=d.error?'block':'none';
    if(d.error)eb.textContent='⚠️ '+d.error;
    const ab=document.getElementById('ab');
    if(d.alert){ab.style.display='block';ab.textContent='✅ '+d.alert_msg;setTimeout(()=>ab.style.display='none',12000);playAlert();}
    document.getElementById('verdict').className='verdict '+(d.take?'take':'skip');
    document.getElementById('vt').textContent=d.take?'✅ TAKE':'❌ SKIP';
    document.getElementById('vd').textContent=d.direction+' · Score '+d.score+'%';
    const bf=document.getElementById('bf');
    document.getElementById('bp').textContent=d.score+'%';
    bf.style.width=d.score+'%';
    bf.style.background=d.take?'var(--green)':'var(--red)';
    const ls=d.score_long||0,ss=d.score_short||0;
    const lv=document.getElementById('lscore'),sv=document.getElementById('sscore');
    lv.textContent=ls+'%'; sv.textContent=ss+'%';
    lv.style.color=ls>=70?'var(--green)':ls>=55?'var(--amber)':'var(--red)';
    sv.style.color=ss>=70?'var(--green)':ss>=55?'var(--amber)':'var(--red)';
    document.getElementById('lcard').style.border=ls>=ss?'2px solid var(--green)':'1px solid var(--border)';
    document.getElementById('scard').style.border=ss>ls?'2px solid var(--green)':'1px solid var(--border)';
    document.getElementById('sl-p').textContent=d.sl_price?d.sl_price.toLocaleString():'—';
    document.getElementById('tp-p').textContent=d.tp_price?d.tp_price.toLocaleString():'—';
    document.getElementById('sl-d').textContent=d.sl_dist?d.sl_dist+' pts':'—';
    document.getElementById('tp-d').textContent=d.sl_dist?(d.sl_dist*2)+' pts (2R)':'—';
    const rc=v=>v>=45&&v<=65?'green':v<35||v>70?'red':'amber';
    document.getElementById('mp').textContent=d.price?.toLocaleString()||'—';
    document.getElementById('mr').className='m-val '+rc(d.rsi); document.getElementById('mr').textContent=d.rsi;
    document.getElementById('me').className='m-val '+(d.ema_diff>0?'green':'red'); document.getElementById('me').textContent=(d.ema_diff>0?'+':'')+d.ema_diff;
    document.getElementById('mv').className='m-val '+(d.vol_ratio>1.2?'green':d.vol_ratio<0.8?'red':'amber'); document.getElementById('mv').textContent=d.vol_ratio+'x';
    document.getElementById('ms').textContent=d.session;
    document.getElementById('mb').className='m-val '+(d.htf_bias==='bullish'?'green':'red'); document.getElementById('mb').textContent=d.htf_bias;
    document.getElementById('pt').textContent='NQ · '+(d.price?.toLocaleString()||'—');
    document.getElementById('factors').innerHTML=(d.factors||[]).map(f=>`<div class="factor ${f.rating}"><div class="f-left"><div class="f-dot"></div><span class="f-name">${f.name}</span></div><span class="f-val">${f.detail}</span></div>`).join('');
    const tot=d.total_trades||0,w=d.wins||0,ls2=tot-w;
    document.getElementById('sw').textContent=w;
    document.getElementById('sl').textContent=ls2;
    const sr=document.getElementById('sr');
    sr.textContent=tot?d.win_rate+'%':'—';
    sr.className='stat-val '+(d.win_rate>=70?'green':d.win_rate>=50?'amber':'red');
    const tl=document.getElementById('tl');
    const trades=(d.trades||[]).slice().reverse().slice(0,10);
    tl.innerHTML=trades.length?trades.map(t=>`<div class="titem ${t.result}"><span style="color:var(--muted)">${(t.time||'').slice(11,16)} ${(t.direction||'').toUpperCase()} @${(t.entry||0).toLocaleString()}</span><span class="tres ${t.result}">${t.result==='win'?'✅ WIN':t.result==='loss'?'❌ LOSS':'⏳ OPEN'} ${t.pnl_pts!=null?(t.pnl_pts>0?'+':'')+t.pnl_pts+'pts':''}</span></div>`).join(''):'<div style="color:var(--muted);font-size:11px;font-family:Courier New,monospace">No trades yet...</div>';
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
      const pe=document.getElementById('at-pnl');
      pe.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+' pts';
      pe.style.color=pnl>=0?'var(--green)':'var(--red)';
    } else { atc.style.display='none'; }
    document.getElementById('ut').textContent='Updated '+d.last_update;
  }catch(e){console.error(e);}
}
function playAlert(){
  try{
    const ctx = new (window.AudioContext||window.webkitAudioContext)();
    [0,150,300].forEach(delay=>{
      const o=ctx.createOscillator();
      const g=ctx.createGain();
      o.connect(g); g.connect(ctx.destination);
      o.frequency.value=880;
      o.type='sine';
      g.gain.setValueAtTime(0,ctx.currentTime+delay/1000);
      g.gain.linearRampToValueAtTime(0.3,ctx.currentTime+delay/1000+0.05);
      g.gain.linearRampToValueAtTime(0,ctx.currentTime+delay/1000+0.3);
      o.start(ctx.currentTime+delay/1000);
      o.stop(ctx.currentTime+delay/1000+0.3);
    });
  }catch(e){}
}

async function closeTrade(r){
  await fetch('/close_trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({result:r})});
  refresh();
}
refresh(); setInterval(refresh,5000);
</script></body></html>"""

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
    data=freq.get_json(); result=data.get("result")
    at=state.get("active_trade")
    if at and result in("win","loss"):
        _close_active(result, at, state.get("price",0))
        return jsonify({"ok":True})
    return jsonify({"ok":False})

if __name__ == "__main__":
    print("\n🚀  IFVG Live AI v3")
    print("📱  Open: http://localhost:5001\n")
    app.run(debug=False, port=5001)
