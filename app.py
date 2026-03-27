"""
app.py  —  IFVG Trade Evaluator Web UI
Run:  python3 app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import sys

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    artefacts      = joblib.load("model.pkl")
    model          = artefacts["model"]
    label_encoders = artefacts["label_encoders"]
    FEATURE_COLS   = artefacts["feature_cols"]
    print("✅  Model loaded successfully")
except FileNotFoundError:
    sys.exit("model.pkl not found — run train_real.py first.")


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
    if rsi <= 35:   d["rsi_zone"] = "oversold"
    elif rsi <= 45: d["rsi_zone"] = "low_neutral"
    elif rsi <= 55: d["rsi_zone"] = "mid_neutral"
    elif rsi <= 65: d["rsi_zone"] = "high_neutral"
    else:           d["rsi_zone"] = "overbought"
    vr = d["volume_ratio"]
    if vr <= 0.8:   d["vol_tier"] = "very_low"
    elif vr <= 1.0: d["vol_tier"] = "low"
    elif vr <= 1.2: d["vol_tier"] = "normal"
    elif vr <= 1.5: d["vol_tier"] = "high"
    else:           d["vol_tier"] = "very_high"
    for col in ["timeframe","session","htf_bias","trade_direction","rsi_zone","vol_tier"]:
        le  = label_encoders[col]
        val = str(d[col])
        d[col + "_enc"] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([d])[FEATURE_COLS]


def get_factors(raw):
    factors = []
    aligned = (
        (raw["htf_bias"]=="bullish" and raw["trade_direction"]=="long") or
        (raw["htf_bias"]=="bearish" and raw["trade_direction"]=="short")
    )
    factors.append(("HTF Bias",
        "good" if aligned else "bad",
        f"{'Aligns' if aligned else 'Conflicts'} with {raw['trade_direction']}"))
    rsi = raw["rsi_at_entry"]
    if 45 <= rsi <= 65:
        factors.append(("RSI", "good", f"{rsi:.1f} — optimal range"))
    elif 35 < rsi < 45 or 65 < rsi < 70:
        factors.append(("RSI", "neutral", f"{rsi:.1f} — acceptable"))
    else:
        factors.append(("RSI", "bad", f"{rsi:.1f} — extreme"))
    ed = raw["ema_diff"]
    aligned_ema = (ed > 0 and raw["trade_direction"]=="long") or (ed < 0 and raw["trade_direction"]=="short")
    factors.append(("EMA Trend",
        "good" if aligned_ema else "bad",
        f"{ed:+.1f} pts {'aligned' if aligned_ema else 'conflicts'}"))
    vr = raw["volume_ratio"]
    if vr > 1.2:   factors.append(("Volume", "good",    f"{vr:.2f}x — strong"))
    elif vr > 0.8: factors.append(("Volume", "neutral", f"{vr:.2f}x — average"))
    else:          factors.append(("Volume", "bad",     f"{vr:.2f}x — weak"))
    sess = raw["session"]
    if sess in ("london","newyork"): factors.append(("Session", "good",    f"{sess.capitalize()} — high liquidity"))
    elif sess == "asia":             factors.append(("Session", "neutral", "Asia — moderate"))
    else:                            factors.append(("Session", "bad",     "Overnight — avoid"))
    sl = raw["sl_distance_points"]
    if sl <= 15:   factors.append(("SL Distance", "good",    f"{sl:.0f} pts — tight"))
    elif sl <= 30: factors.append(("SL Distance", "neutral", f"{sl:.0f} pts — ok"))
    else:          factors.append(("SL Distance", "bad",     f"{sl:.0f} pts — wide"))
    return factors


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IFVG Trade Evaluator</title>

<style>
  :root {
    --bg:       #0a0c0f;
    --surface:  #111318;
    --border:   #1e2229;
    --text:     #e8eaf0;
    --muted:    #5a6070;
    --green:    #00e5a0;
    --red:      #ff4d6d;
    --amber:    #ffb627;
    --blue:     #4d9fff;
    --accent:   #00e5a0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    padding: 24px 16px;
  }
  .wrap { max-width: 480px; margin: 0 auto; }

  /* Header */
  header { margin-bottom: 32px; }
  .logo { font-size: 11px; font-family: 'Courier New', monospace; color: var(--muted); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px; }
  h1 { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
  h1 span { color: var(--accent); }

  /* Card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 10px; font-family: 'Courier New', monospace;
    color: var(--muted); letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 20px;
  }

  /* Form grid */
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field.full { grid-column: 1 / -1; }
  label { font-size: 10px; font-family: 'Courier New', monospace; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; }

  input, select {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'Courier New', monospace;
    font-size: 13px;
    padding: 10px 12px;
    width: 100%;
    outline: none;
    transition: border-color 0.2s;
    -webkit-appearance: none;
  }
  input:focus, select:focus { border-color: var(--accent); }

  /* Toggle buttons for direction */
  .toggle-group { display: flex; gap: 8px; }
  .toggle {
    flex: 1; padding: 10px; border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg); color: var(--muted);
    font-family: 'Courier New', monospace; font-size: 12px;
    cursor: pointer; text-align: center; transition: all 0.15s;
    letter-spacing: 1px;
  }
  .toggle.active-long  { background: rgba(0,229,160,0.12); border-color: var(--green); color: var(--green); }
  .toggle.active-short { background: rgba(255,77,109,0.12); border-color: var(--red);   color: var(--red);   }

  /* Submit */
  .btn {
    width: 100%; padding: 14px;
    background: var(--accent); color: #000;
    border: none; border-radius: 10px;
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 15px; font-weight: 800;
    cursor: pointer; letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.1s;
    margin-top: 4px;
  }
  .btn:hover { opacity: 0.9; }
  .btn:active { transform: scale(0.99); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Result */
  #result { display: none; }
  .verdict {
    border-radius: 12px; padding: 24px;
    margin-bottom: 16px; text-align: center;
  }
  .verdict.take { background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.3); }
  .verdict.skip { background: rgba(255,77,109,0.08); border: 1px solid rgba(255,77,109,0.3); }
  .verdict-label { font-size: 11px; font-family: 'Courier New', monospace; color: var(--muted); letter-spacing: 2px; margin-bottom: 8px; }
  .verdict-text { font-size: 32px; font-weight: 800; letter-spacing: -1px; }
  .verdict.take .verdict-text { color: var(--green); }
  .verdict.skip .verdict-text { color: var(--red); }
  .verdict-sub { font-size: 13px; color: var(--muted); margin-top: 4px; font-family: 'Courier New', monospace; }

  /* Probability bar */
  .prob-wrap { margin: 16px 0 4px; }
  .prob-label { display: flex; justify-content: space-between; font-size: 11px; font-family: 'Courier New', monospace; color: var(--muted); margin-bottom: 8px; }
  .prob-track { height: 6px; background: var(--border); border-radius: 99px; overflow: hidden; }
  .prob-fill { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }

  /* Factors */
  .factor-list { display: flex; flex-direction: column; gap: 8px; }
  .factor {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-radius: 8px;
    background: var(--bg); border: 1px solid var(--border);
  }
  .factor-name { font-size: 12px; font-family: 'Courier New', monospace; color: var(--muted); }
  .factor-detail { font-size: 12px; text-align: right; max-width: 60%; }
  .factor.good  .factor-detail { color: var(--green); }
  .factor.neutral .factor-detail { color: var(--amber); }
  .factor.bad   .factor-detail { color: var(--red); }
  .factor-dot { width: 6px; height: 6px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
  .factor.good    .factor-dot { background: var(--green); }
  .factor.neutral .factor-dot { background: var(--amber); }
  .factor.bad     .factor-dot { background: var(--red); }
  .factor-left { display: flex; align-items: center; }

  /* Loading */
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(0,0,0,0.3); border-top-color: #000; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Footer */
  footer { text-align: center; margin-top: 32px; font-size: 11px; font-family: 'Courier New', monospace; color: var(--border); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">NQ / MNQ Futures</div>
    <h1>IFVG <span>Evaluator</span></h1>
  </header>

  <!-- Paste Card -->
  <div class="card">
    <div class="card-title">Quick Paste from Chart</div>
    <div class="field">
      <label>Paste signal data from TradingView</label>
      <input type="text" id="paste_input" placeholder="📋 DIR:long RSI:52.3 EMA:6.1 VOL:1.4 SESS:newyork BIAS:bullish SL:18.5 SCORE:71%" style="font-size:11px;">
    </div>
    <br>
    <button class="btn" onclick="parsePaste()" style="background:#4d9fff;">⚡ PARSE & EVALUATE</button>
  </div>

  <!-- Input Card -->
  <div class="card">
    <div class="card-title">Manual Input</div>
    <div class="grid">

      <div class="field full">
        <label>Direction</label>
        <div class="toggle-group">
          <div class="toggle active-long" id="btn-long"  onclick="setDir('long')">▲ LONG</div>
          <div class="toggle"             id="btn-short" onclick="setDir('short')">▼ SHORT</div>
        </div>
        <input type="hidden" id="trade_direction" value="long">
      </div>

      <div class="field">
        <label>Session</label>
        <select id="session">
          <option value="newyork" selected>New York</option>
          <option value="london">London</option>
          <option value="asia">Asia</option>
          <option value="overnight">Overnight</option>
        </select>
      </div>

      <div class="field">
        <label>HTF Bias</label>
        <select id="htf_bias">
          <option value="bullish" selected>Bullish</option>
          <option value="bearish">Bearish</option>
        </select>
      </div>

      <div class="field">
        <label>RSI at Entry</label>
        <input type="number" id="rsi_at_entry" value="52" min="1" max="99" step="0.1">
      </div>

      <div class="field">
        <label>EMA Diff (pts)</label>
        <input type="number" id="ema_diff" value="6" step="0.1" placeholder="EMA9 - EMA21">
      </div>

      <div class="field">
        <label>Volume Ratio</label>
        <input type="number" id="volume_ratio" value="1.4" min="0.1" max="10" step="0.05">
      </div>

      <div class="field">
        <label>SL Distance (pts)</label>
        <input type="number" id="sl_distance_points" value="20" min="1" step="0.5">
      </div>

      <div class="field full">
        <label>Entry Price</label>
        <input type="number" id="entry_price" value="21000" step="0.25">
      </div>

    </div>
    <br>
    <button class="btn" id="evalBtn" onclick="evaluate()">EVALUATE SETUP</button>
  </div>

  <!-- Result -->
  <div id="result">
    <div class="verdict" id="verdict-box">
      <div class="verdict-label">AI RECOMMENDATION</div>
      <div class="verdict-text" id="verdict-text"></div>
      <div class="verdict-sub" id="verdict-sub"></div>
      <div class="prob-wrap">
        <div class="prob-label">
          <span>WIN PROBABILITY</span>
          <span id="prob-pct"></span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" id="prob-fill" style="width:0%"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Factor Breakdown</div>
      <div class="factor-list" id="factors"></div>
    </div>
  </div>

  <footer>IFVG Trade Evaluator · Trained on real NQ data</footer>
</div>

<script>
function setDir(dir) {
  document.getElementById('trade_direction').value = dir;
  document.getElementById('btn-long').className  = 'toggle' + (dir==='long'  ? ' active-long'  : '');
  document.getElementById('btn-short').className = 'toggle' + (dir==='short' ? ' active-short' : '');
}

async function evaluate() {
  const btn = document.getElementById('evalBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>EVALUATING...';

  const payload = {
    timeframe:          "1H",
    rsi_at_entry:       parseFloat(document.getElementById('rsi_at_entry').value),
    ema_diff:           parseFloat(document.getElementById('ema_diff').value),
    volume_ratio:       parseFloat(document.getElementById('volume_ratio').value),
    session:            document.getElementById('session').value,
    htf_bias:           document.getElementById('htf_bias').value,
    trade_direction:    document.getElementById('trade_direction').value,
    sl_distance_points: parseFloat(document.getElementById('sl_distance_points').value),
    entry_price:        parseFloat(document.getElementById('entry_price').value),
  };

  try {
    const res  = await fetch('/evaluate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();

    // Verdict
    const box = document.getElementById('verdict-box');
    box.className = 'verdict ' + (data.take ? 'take' : 'skip');
    document.getElementById('verdict-text').textContent = data.take ? (data.strong ? '✅ TAKE — STRONG' : '✅ TAKE') : '❌ SKIP';
    document.getElementById('verdict-sub').textContent  = `Win probability ${data.prob_pct}`;

    // Bar
    const pct = data.prob_raw * 100;
    document.getElementById('prob-pct').textContent = data.prob_pct;
    const fill = document.getElementById('prob-fill');
    fill.style.background = data.take ? 'var(--green)' : 'var(--red)';
    setTimeout(() => fill.style.width = pct + '%', 50);

    // Factors
    const container = document.getElementById('factors');
    container.innerHTML = data.factors.map(f => `
      <div class="factor ${f.rating}">
        <div class="factor-left">
          <div class="factor-dot"></div>
          <span class="factor-name">${f.name}</span>
        </div>
        <span class="factor-detail">${f.detail}</span>
      </div>`).join('');

    document.getElementById('result').style.display = 'block';
    document.getElementById('result').scrollIntoView({ behavior: 'smooth' });

  } catch(e) {
    alert('Error: ' + e.message);
  }

  btn.disabled = false;
  btn.innerHTML = 'EVALUATE SETUP';
}

function parsePaste() {
  const raw = document.getElementById('paste_input').value;
  if (!raw) { alert('Paste the signal data from TradingView first'); return; }
  
  const get = (key) => {
    const match = raw.match(new RegExp(key + ':([^\\s]+)'));
    return match ? match[1] : null;
  };
  
  const dir  = get('DIR');
  const rsi  = get('RSI');
  const ema  = get('EMA');
  const vol  = get('VOL');
  const sess = get('SESS');
  const bias = get('BIAS');
  const sl   = get('SL');
  
  if (!dir || !rsi) { alert('Could not parse data. Make sure you copied the full 📋 line from TradingView.'); return; }
  
  // Fill the form
  setDir(dir);
  document.getElementById('rsi_at_entry').value       = rsi;
  document.getElementById('ema_diff').value           = ema || '0';
  document.getElementById('volume_ratio').value       = vol || '1';
  document.getElementById('session').value            = sess || 'newyork';
  document.getElementById('htf_bias').value           = bias || 'bullish';
  document.getElementById('sl_distance_points').value = sl || '20';
  
  // Auto evaluate
  evaluate();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    raw = request.get_json()
    X   = engineer(raw)
    prob_win = float(model.predict_proba(X)[0, 1])
    take   = prob_win >= 0.58
    strong = prob_win >= 0.70
    factors = []
    for name, rating, detail in get_factors(raw):
        factors.append({"name": name, "rating": rating, "detail": detail})
    return jsonify({
        "prob_raw": round(prob_win, 4),
        "prob_pct": f"{prob_win:.1%}",
        "take":     take,
        "strong":   strong,
        "factors":  factors,
    })

if __name__ == "__main__":
    print("\n🚀  Starting IFVG Trade Evaluator...")
    print("📱  Open in browser: http://localhost:5000\n")
    app.run(debug=False, port=5000)
