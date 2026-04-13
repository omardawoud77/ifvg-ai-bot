"""
live_app.py — ICT Strict Filter Bot v2
Based on backtest_v2.py — all 5 filters must pass simultaneously
Filters: Volume 1.5x, HTF alignment (W=D=4H), 1H momentum, RSI 35-65, FVG or OB
Trade persistence: writes to trades_log.json on every close (survives restarts)
"""

from flask import Flask, jsonify, render_template_string
import numpy as np
import pandas as pd
import requests as req
from datetime import datetime, timezone
import json, os, time, threading, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RR         = 1.0
VOL_MULT   = 1.5
MAX_DAILY  = 5
COOLDOWN   = 6       # bars (5m each = 30 min)
INTERVAL   = 15      # seconds between fetches
TRADE_LOG  = "trades_log.json"
CAPITAL    = 10000.0
PNL_PER_PT = 2.0     # MNQ: $2/point

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "price": 0.0, "direction": "—", "session": "—",
    "rsi": 50.0, "vol_ratio": 0.0, "htf": "—",
    "sl_price": 0.0, "tp_price": 0.0,
    "filters": {},
    "take": False, "active_trade": None,
    "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
    "capital": CAPITAL, "pnl": 0.0,
    "last_update": "—", "error": None,
    "trading_paused": False,
}

bars_buffer = []
last_bar_time = None
last_trade_bar = -999
daily_counts = {}
_lock = threading.Lock()

# ── Trade persistence ─────────────────────────────────────────────────────────
def load_trades():
    if os.path.exists(TRADE_LOG):
        try:
            with open(TRADE_LOG) as f:
                return json.load(f)
        except:
            return []
    return []

def save_trade(trade):
    trades = load_trades()
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)

def get_stats():
    trades = load_trades()
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    wins = [t for t in closed if t["result"] == "win"]
    total_pnl = sum(t["pnl_pts"] for t in closed) * PNL_PER_PT
    wr = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    return len(closed), len(wins), len(closed) - len(wins), wr, total_pnl

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def calc_rsi(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, 0.001)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def get_mtf(closes):
    """EMA-based MTF bias — same logic as backtest_v2"""
    c = pd.Series(closes)
    weekly = "bull" if calc_ema(c, 576).iloc[-1] > calc_ema(c, 1440).iloc[-1] else "bear"
    daily  = "bull" if calc_ema(c, 288).iloc[-1] > calc_ema(c, 576).iloc[-1]  else "bear"
    h4     = "bull" if calc_ema(c, 48).iloc[-1]  > calc_ema(c, 96).iloc[-1]   else "bear"
    h1     = "bull" if calc_ema(c, 12).iloc[-1]  > calc_ema(c, 24).iloc[-1]   else "bear"
    return weekly, daily, h4, h1

def get_session(dt):
    h = dt.hour * 60 + dt.minute
    if 360  <= h < 480:  return "london_open", True
    if 480  <= h < 720:  return "london",      True
    if 780  <= h < 960:  return "ny_open",     True
    if 960  <= h < 1080: return "ny_pm",       True
    return "other", False

def find_fvg(df_slice, direction):
    for i in range(2, min(30, len(df_slice) - 1)):
        c1 = df_slice.iloc[-i - 1]
        c3 = df_slice.iloc[-i + 1]
        if direction == "long":
            lo, hi = float(c1["high"]), float(c3["low"])
            if hi > lo:
                return lo, hi
        else:
            hi, lo = float(c1["low"]), float(c3["high"])
            if hi < lo:
                return lo, hi
    return None, None

def find_ob(df_slice, direction):
    avg_range = float((df_slice.tail(20)["high"] - df_slice.tail(20)["low"]).mean())
    for i in range(3, min(30, len(df_slice))):
        bar = df_slice.iloc[-i]
        br = float(bar["high"]) - float(bar["low"])
        if br > avg_range * 1.5:
            return float(bar["low"]), float(bar["high"])
    return None, None

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_bars():
    global bars_buffer, last_bar_time
    url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval=5m&range=5d"
    r = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    js = r.json()
    res = js["chart"]["result"][0]
    ts  = res["timestamp"]
    q   = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "close":  q["close"],
        "high":   q["high"],
        "low":    q["low"],
        "open":   q["open"],
        "volume": q["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()

    if df.empty or len(df) < 50:
        return None

    # Only update if we have a new bar
    latest = df.index[-1]
    if last_bar_time is not None and latest == last_bar_time:
        return None  # no new bar

    last_bar_time = latest
    bars_buffer = df
    return df

# ── Check active trade ────────────────────────────────────────────────────────
def check_active_trade(df):
    at = state.get("active_trade")
    if not at or at["result"] != "open":
        return

    cur_high = float(df.iloc[-1]["high"])
    cur_low  = float(df.iloc[-1]["low"])
    price    = float(df.iloc[-1]["close"])

    if at["direction"] == "long":
        if cur_low  <= at["sl"]: _close_trade("loss", at, at["sl"])
        elif cur_high >= at["tp"]: _close_trade("win",  at, at["tp"])
    else:
        if cur_high >= at["sl"]: _close_trade("loss", at, at["sl"])
        elif cur_low  <= at["tp"]: _close_trade("win",  at, at["tp"])

def _close_trade(result, at, exit_price):
    at["result"]     = result
    at["exit_price"] = round(exit_price, 2)
    at["exit_time"]  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    pnl_pts = round(abs(at["tp"] - at["entry"]) if result == "win" else -abs(at["sl"] - at["entry"]), 1)
    at["pnl_pts"] = pnl_pts
    at["pnl_usd"] = round(pnl_pts * PNL_PER_PT, 2)

    save_trade(at)
    state["active_trade"] = None

    total, wins, losses, wr, total_pnl = get_stats()
    state.update({
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": wr, "capital": round(CAPITAL + total_pnl, 2),
        "pnl": round(total_pnl, 2),
    })

    emoji = "✅" if result == "win" else "❌"
    log.info(f"{emoji} CLOSED {result.upper()} | PnL: ${at['pnl_usd']:+.0f} ({pnl_pts:+.1f}pts) | "
             f"Bars: — | Capital: ${state['capital']} | WR: {wr}% ({wins}W/{losses}L)")

# ── Main signal logic ─────────────────────────────────────────────────────────
def run_signal(df):
    global last_trade_bar, daily_counts

    if state.get("trading_paused"):
        return

    i = len(df) - 1
    bar = df.iloc[-1]
    bt  = df.index[-1].to_pydatetime()

    # Session filter
    session, active = get_session(bt)
    state["session"] = session
    if not active:
        state["filters"] = {"session": False}
        state["take"] = False
        return

    # Daily cap
    day = bt.strftime("%Y-%m-%d")
    if daily_counts.get(day, 0) >= MAX_DAILY:
        state["filters"] = {"daily_cap": False}
        state["take"] = False
        return

    # Cooldown
    if i - last_trade_bar < COOLDOWN:
        state["filters"] = {"cooldown": False}
        state["take"] = False
        return

    sl = df.tail(50)
    filters = {}

    # Filter 1: Volume spike
    vol     = float(bar.get("volume", 0))
    avg_vol = float(sl["volume"].tail(20).mean())
    vol_ratio = round(vol / avg_vol, 2) if avg_vol > 0 else 0.0
    filters["volume"] = vol_ratio >= VOL_MULT
    state["vol_ratio"] = vol_ratio
    if not filters["volume"]:
        state["filters"] = filters
        state["take"] = False
        return

    # Filter 2: HTF alignment
    closes = list(df["close"])
    if len(closes) < 600:
        state["take"] = False
        return
    weekly, daily_bias, h4, h1 = get_mtf(closes)
    filters["htf_aligned"] = (weekly == daily_bias == h4)
    state["htf"] = f"{weekly}/{daily_bias}/{h4}/{h1}"
    if not filters["htf_aligned"]:
        state["filters"] = filters
        state["take"] = False
        return

    direction = "long" if h4 == "bull" else "short"

    # Filter 3: 1H momentum
    filters["momentum"] = (direction == "long" and h1 == "bull") or (direction == "short" and h1 == "bear")
    if not filters["momentum"]:
        state["filters"] = filters
        state["take"] = False
        return

    # Filter 4: RSI
    rsi = calc_rsi(sl["close"])
    filters["rsi"] = not (direction == "long" and rsi > 65) and not (direction == "short" and rsi < 35)
    state["rsi"] = round(rsi, 1)
    if not filters["rsi"]:
        state["filters"] = filters
        state["take"] = False
        return

    # Filter 5: FVG or OB
    close = float(bar["close"])
    fvg_lo, fvg_hi = find_fvg(sl, direction)
    ob_lo,  ob_hi  = find_ob(sl, direction)
    at_fvg = fvg_lo is not None and fvg_lo <= close <= fvg_hi
    at_ob  = ob_lo  is not None and ob_lo  <= close <= ob_hi
    filters["fvg_or_ob"] = at_fvg or at_ob
    if not filters["fvg_or_ob"]:
        state["filters"] = filters
        state["take"] = False
        return

    # ALL FILTERS PASSED
    filters["all_pass"] = True
    state["filters"] = filters

    atr     = float((df.tail(20)["high"] - df.tail(20)["low"]).mean())
    sl_dist = max(20.0, min(50.0, atr))

    if direction == "long":
        sl_price = round(close - sl_dist, 2)
        tp_price = round(close + sl_dist * RR, 2)
    else:
        sl_price = round(close + sl_dist, 2)
        tp_price = round(close - sl_dist * RR, 2)

    state.update({
        "price": round(close, 2),
        "direction": direction.upper(),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "take": True,
    })

    # Only enter if no active trade
    if not state.get("active_trade"):
        trade = {
            "time":      bt.strftime("%Y-%m-%d %H:%M:%S"),
            "session":   session,
            "direction": direction,
            "entry":     round(close, 2),
            "sl":        sl_price,
            "tp":        tp_price,
            "sl_dist":   round(sl_dist, 1),
            "vol_ratio": vol_ratio,
            "rsi":       round(rsi, 1),
            "at_fvg":    at_fvg,
            "at_ob":     at_ob,
            "htf":       f"{weekly}/{daily_bias}/{h4}/{h1}",
            "result":    "open",
            "exit_price": None,
            "exit_time":  None,
            "pnl_pts":    None,
            "pnl_usd":    None,
        }
        state["active_trade"] = trade
        last_trade_bar = i
        daily_counts[day] = daily_counts.get(day, 0) + 1

        emoji = "📈" if direction == "long" else "📉"
        log.info(f"{emoji} PAPER {direction.upper()} @ {close:.2f} | "
                 f"SL: {sl_price} TP: {tp_price} | Session: {session} | "
                 f"Vol: {vol_ratio:.2f}x RSI: {rsi:.1f} HTF: {weekly}/{daily_bias}/{h4}/{h1}")

# ── Background loop ───────────────────────────────────────────────────────────
def trading_loop():
    log.info("🚀 ICT Strict Filter Bot v2 started")
    # Load persisted stats on startup
    total, wins, losses, wr, total_pnl = get_stats()
    state.update({
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": wr, "capital": round(CAPITAL + total_pnl, 2),
        "pnl": round(total_pnl, 2),
    })
    log.info(f"📊 Loaded {total} historical trades | WR: {wr}% | Capital: ${state['capital']}")

    bar_count = 0
    while True:
        try:
            log.info("📥 Fetching latest bars...")
            df = fetch_bars()
            if df is None:
                log.info("⏸️  No new bars — waiting 15s")
                time.sleep(INTERVAL)
                continue

            bar_count += 1
            log.info(f"📊 {bar_count} new bar(s) received")

            with _lock:
                check_active_trade(df)
                if not state.get("active_trade"):
                    run_signal(df)

                state["price"]       = round(float(df.iloc[-1]["close"]), 2)
                state["last_update"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
                state["error"]       = None

            # Status log
            at = state.get("active_trade")
            pos = f"{at['direction'].upper()} @ {at['entry']}" if at else "FLAT"
            log.info(f"Bar {bar_count} | Pos: {pos} | Capital: ${state['capital']} | "
                     f"WR: {state['win_rate']}% | Trades: {state['total_trades']}")

        except Exception as e:
            import traceback
            state["error"] = str(e)
            log.error(f"Error: {e}")
            log.error(traceback.format_exc())

        time.sleep(INTERVAL)

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    with _lock:
        s = dict(state)
    at = s.get("active_trade")
    trades = load_trades()
    closed = [t for t in trades if t["result"] in ("win", "loss")][-20:]  # last 20

    rows = ""
    for t in reversed(closed):
        color = "#00e5a0" if t["result"] == "win" else "#ff4d6d"
        pnl = t.get("pnl_usd", 0) or 0
        rows += f"""<tr>
            <td>{t['time'][:16]}</td>
            <td>{t['direction'].upper()}</td>
            <td>{t['session']}</td>
            <td>{t['entry']}</td>
            <td>{t.get('exit_price','—')}</td>
            <td style="color:{color}">${pnl:+.0f}</td>
            <td style="color:{color}">{t['result'].upper()}</td>
        </tr>"""

    filters = s.get("filters", {})
    filter_html = ""
    filter_map = {
        "session": "Session kill zone",
        "volume": f"Volume ≥1.5x ({s['vol_ratio']:.2f}x)",
        "htf_aligned": f"HTF aligned ({s['htf']})",
        "momentum": "1H momentum",
        "rsi": f"RSI 35–65 ({s['rsi']:.1f})",
        "fvg_or_ob": "FVG or OB",
    }
    for k, label in filter_map.items():
        val = filters.get(k)
        if val is True:
            dot = "🟢"
        elif val is False:
            dot = "🔴"
        else:
            dot = "⚪"
        filter_html += f"<div class='frow'><span>{dot}</span><span>{label}</span></div>"

    active_html = ""
    if at and at["result"] == "open":
        active_html = f"""
        <div class='active-trade'>
            <div class='at-title'>⚡ OPEN TRADE</div>
            <div class='at-grid'>
                <span>Dir: <b>{at['direction'].upper()}</b></span>
                <span>Entry: <b>{at['entry']}</b></span>
                <span>SL: <b style="color:#ff4d6d">{at['sl']}</b></span>
                <span>TP: <b style="color:#00e5a0">{at['tp']}</b></span>
            </div>
        </div>"""

    return render_template_string(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICT Bot v2</title>
<meta http-equiv="refresh" content="15">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0c0f;color:#e8eaf0;font-family:-apple-system,sans-serif;padding:16px}}
.wrap{{max-width:600px;margin:0 auto}}
h1{{font-size:20px;font-weight:700;margin-bottom:16px;color:#00e5a0}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px}}
.card{{background:#111318;border:1px solid #1e2229;border-radius:10px;padding:14px;text-align:center}}
.lbl{{font-size:10px;color:#5a6070;margin-bottom:4px;letter-spacing:1px}}
.val{{font-size:22px;font-weight:700}}
.green{{color:#00e5a0}}.red{{color:#ff4d6d}}.amber{{color:#ffb627}}
.filters{{background:#111318;border:1px solid #1e2229;border-radius:10px;padding:14px;margin-bottom:16px}}
.ftitle{{font-size:11px;color:#5a6070;letter-spacing:2px;margin-bottom:10px}}
.frow{{display:flex;align-items:center;gap:8px;font-size:13px;padding:4px 0}}
.active-trade{{background:rgba(255,182,39,.1);border:1px solid rgba(255,182,39,.4);border-radius:10px;padding:14px;margin-bottom:16px}}
.at-title{{font-size:11px;color:#ffb627;letter-spacing:2px;margin-bottom:8px}}
.at-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:13px}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:#111318;border-radius:10px;overflow:hidden}}
th{{background:#1e2229;padding:8px;text-align:left;color:#5a6070;font-weight:500}}
td{{padding:8px;border-bottom:1px solid #1e2229}}
.error{{background:rgba(255,77,109,.1);border:1px solid #ff4d6d;border-radius:8px;padding:10px;color:#ff4d6d;margin-bottom:12px;font-size:12px}}
</style></head>
<body><div class="wrap">
<h1>⚡ ICT Strict Filter Bot v2</h1>
{"<div class='error'>⚠️ "+s['error']+"</div>" if s.get('error') else ""}
<div class="grid">
    <div class="card"><div class="lbl">CAPITAL</div><div class="val {'green' if s['pnl']>=0 else 'red'}">${s['capital']:,.0f}</div></div>
    <div class="card"><div class="lbl">WIN RATE</div><div class="val amber">{s['win_rate']}%</div></div>
    <div class="card"><div class="lbl">TRADES</div><div class="val">{s['total_trades']}</div></div>
    <div class="card"><div class="lbl">P&L</div><div class="val {'green' if s['pnl']>=0 else 'red'}">${s['pnl']:+,.0f}</div></div>
    <div class="card"><div class="lbl">WINS/LOSSES</div><div class="val">{s['wins']}W / {s['losses']}L</div></div>
    <div class="card"><div class="lbl">PRICE</div><div class="val">{s['price']:,.1f}</div></div>
</div>
{active_html}
<div class="filters">
    <div class="ftitle">FILTER STATUS — {s['session'].upper()} — {s['last_update']}</div>
    {filter_html}
</div>
<table>
    <tr><th>Time</th><th>Dir</th><th>Session</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Result</th></tr>
    {rows if rows else "<tr><td colspan='7' style='text-align:center;color:#5a6070;padding:20px'>No closed trades yet</td></tr>"}
</table>
<div style="font-size:10px;color:#5a6070;margin-top:10px;text-align:center">
    Auto-refresh 15s · ICT Strict v2 · RR:{RR} Vol:{VOL_MULT}x MaxDaily:{MAX_DAILY}
</div>
</div></body></html>""")


@app.route("/state")
def get_state():
    with _lock:
        s = dict(state)
    trades = load_trades()
    s["trades"] = trades
    s["total_trades"] = len([t for t in trades if t["result"] in ("win","loss")])
    # Convert numpy types for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj
    return jsonify({k: clean(v) if not isinstance(v, (dict, list)) else v for k, v in s.items()})


@app.route("/trades")
def get_trades():
    trades = load_trades()
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    wins   = [t for t in closed if t["result"] == "win"]
    total_pnl = sum(t.get("pnl_pts", 0) or 0 for t in closed) * PNL_PER_PT
    wr = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    return jsonify({
        "total": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate": wr,
        "total_pnl_usd": round(total_pnl, 2),
        "capital": round(CAPITAL + total_pnl, 2),
        "trades": trades,
    })


@app.route("/pause", methods=["POST"])
def pause():
    state["trading_paused"] = True
    log.info("⏸️  Trading paused")
    return jsonify({"status": "paused"})


@app.route("/resume", methods=["POST"])
def resume():
    state["trading_paused"] = False
    log.info("▶️  Trading resumed")
    return jsonify({"status": "resumed"})


@app.route("/close_trade", methods=["POST"])
def close_trade_manual():
    with _lock:
        at = state.get("active_trade")
        if not at:
            return jsonify({"error": "no active trade"}), 400
        price = state.get("price", at["entry"])
        pnl_pts = round(price - at["entry"] if at["direction"] == "long" else at["entry"] - price, 1)
        at["result"] = "win" if pnl_pts > 0 else "loss"
        at["exit_price"] = price
        at["exit_time"]  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        at["pnl_pts"] = pnl_pts
        at["pnl_usd"] = round(pnl_pts * PNL_PER_PT, 2)
        save_trade(at)
        state["active_trade"] = None
        total, wins, losses, wr, total_pnl = get_stats()
        state.update({
            "total_trades": total, "wins": wins, "losses": losses,
            "win_rate": wr, "capital": round(CAPITAL + total_pnl, 2),
            "pnl": round(total_pnl, 2),
        })
    return jsonify({"status": "closed", "pnl_pts": pnl_pts})


@app.route("/reset_trades", methods=["POST"])
def reset_trades():
    """Clear all trades for a fresh start"""
    with open(TRADE_LOG, "w") as f:
        json.dump([], f)
    state.update({
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "capital": CAPITAL, "pnl": 0.0,
        "active_trade": None,
    })
    log.info("🔄 Trades reset")
    return jsonify({"status": "reset"})


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
