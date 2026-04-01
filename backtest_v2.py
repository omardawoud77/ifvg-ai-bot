import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("📥 Downloading 30 days of NQ 5m data...")
df = yf.download("NQ=F", period="30d", interval="5m", progress=False)
df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
df = df.dropna()
print(f"✅ Got {len(df)} bars")

def get_mtf(df_full, idx):
    try:
        s = df_full.iloc[max(0,idx-600):idx+1]
        c = s['close']
        def ema(x,n): return x.ewm(span=n).mean()
        weekly = 'bull' if ema(c,576).iloc[-1] > ema(c,1440).iloc[-1] else 'bear'
        daily  = 'bull' if ema(c,288).iloc[-1] > ema(c,576).iloc[-1] else 'bear'
        h4     = 'bull' if ema(c,48).iloc[-1]  > ema(c,96).iloc[-1]  else 'bear'
        h1     = 'bull' if ema(c,12).iloc[-1]  > ema(c,24).iloc[-1]  else 'bear'
        return weekly, daily, h4, h1
    except:
        return '','','',''

def get_session(dt):
    h = dt.hour * 60 + dt.minute
    if 360 <= h < 480:  return "london_open", True
    if 480 <= h < 720:  return "london", True
    if 780 <= h < 960:  return "ny_open", True
    if 960 <= h < 1080: return "ny_pm", True
    return "other", False  # block asia + transitions

def calc_rsi(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return float((100 - 100/(1+g/l.replace(0,0.001))).iloc[-1])

def find_fvg(df_slice, direction):
    """Find most recent unfilled FVG in the direction."""
    for i in range(2, min(30, len(df_slice)-1)):
        c1 = df_slice.iloc[-i-1]
        c3 = df_slice.iloc[-i+1]
        if direction == 'long':
            gap_lo = float(c1['high'])
            gap_hi = float(c3['low'])
            if gap_hi > gap_lo:
                return gap_lo, gap_hi
        else:
            gap_hi = float(c1['low'])
            gap_lo = float(c3['high'])
            if gap_hi < gap_lo:
                return gap_lo, gap_hi
    return None, None

def find_ob(df_slice, direction):
    """Find nearest Order Block — last large candle before a move."""
    avg_range = float((df_slice.tail(20)['high']-df_slice.tail(20)['low']).mean())
    for i in range(3, min(30, len(df_slice))):
        bar = df_slice.iloc[-i]
        br = float(bar['high']) - float(bar['low'])
        if br > avg_range * 1.5:
            return float(bar['low']), float(bar['high'])
    return None, None

# ── Main backtest ────────────────────────────────────────────────────────
RR = 1.0  # 1RR everywhere as requested
MAX_DAILY = 5
COOLDOWN = 6   # 6 bars = 30 min
VOL_MULT = 1.5  # volume must be 2.5x average

trades = []
last_bar = -999
daily = {}

print("🔄 Running v2 backtest (strict filters)...")

for i in range(300, len(df)):
    if i - last_bar < COOLDOWN:
        continue

    bar = df.iloc[i]
    bt = df.index[i]
    session, active = get_session(bt)
    if not active:
        continue

    day = bt.strftime("%Y-%m-%d")
    if daily.get(day,0) >= MAX_DAILY:
        continue

    # ── FILTER 1: Volume spike ───────────────────────────────────────────
    sl = df.iloc[max(0,i-50):i+1]
    vol = float(bar.get('volume', 0))
    avg_vol = float(sl['volume'].tail(20).mean())
    if avg_vol == 0 or vol < avg_vol * VOL_MULT:
        continue

    # ── FILTER 2: HTF alignment ──────────────────────────────────────────
    weekly, daily_bias, h4, h1 = get_mtf(df, i)
    if not (weekly and daily_bias and h4 and h1):
        continue
    if not (weekly == daily_bias == h4):
        continue  # all 3 must agree
    direction = 'long' if h4 == 'bull' else 'short'

    # ── FILTER 3: 1H momentum confirms ──────────────────────────────────
    if direction == 'long' and h1 != 'bull':
        continue
    if direction == 'short' and h1 != 'bear':
        continue

    # ── FILTER 4: RSI not overextended ──────────────────────────────────
    rsi = calc_rsi(sl['close'])
    if direction == 'long' and rsi > 65:
        continue
    if direction == 'short' and rsi < 35:
        continue

    # ── FILTER 5: Price at FVG or OB ────────────────────────────────────
    close = float(bar['close'])
    fvg_lo, fvg_hi = find_fvg(sl, direction)
    ob_lo, ob_hi = find_ob(sl, direction)

    at_fvg = fvg_lo is not None and fvg_lo <= close <= fvg_hi
    at_ob  = ob_lo is not None and ob_lo <= close <= ob_hi

    if not (at_fvg or at_ob):
        continue

    # ── Entry ────────────────────────────────────────────────────────────
    entry = close
    atr = float((df.iloc[max(0,i-20):i]['high']-df.iloc[max(0,i-20):i]['low']).mean())
    sl_dist = max(20.0, min(50.0, atr * 1.0))

    if direction == 'long':
        sl_price = entry - sl_dist
        tp_price = entry + sl_dist * RR
    else:
        sl_price = entry + sl_dist
        tp_price = entry - sl_dist * RR

    # ── Simulate ─────────────────────────────────────────────────────────
    result = None
    exit_p = None
    for j in range(i+1, min(i+120, len(df))):
        f = df.iloc[j]
        if direction == 'long':
            if float(f['low']) <= sl_price:
                result='loss'; exit_p=sl_price; break
            if float(f['high']) >= tp_price:
                result='win'; exit_p=tp_price; break
        else:
            if float(f['high']) >= sl_price:
                result='loss'; exit_p=sl_price; break
            if float(f['low']) <= tp_price:
                result='win'; exit_p=tp_price; break

    if not result:
        result='timeout'; exit_p=float(df.iloc[min(i+119,len(df)-1)]['close'])

    pnl = (exit_p-entry) if direction=='long' else (entry-exit_p)

    trades.append({
        'time': str(bt), 'session': session,
        'direction': direction, 'entry': round(entry,2),
        'sl': round(sl_price,2), 'tp': round(tp_price,2),
        'exit': round(exit_p,2), 'pnl': round(pnl,1),
        'result': result, 'vol_ratio': round(vol/avg_vol,2),
        'rsi': round(rsi,1), 'at_fvg': at_fvg, 'at_ob': at_ob,
        'sl_dist': round(sl_dist,1), 'htf': f"{weekly}/{daily_bias}/{h4}/{h1}"
    })

    last_bar = i
    daily[day] = daily.get(day,0) + 1

# ── Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BACKTEST v2 — STRICT FILTERS — Last 30 Days")
print(f"{'='*60}")

total = len(trades)
wins = [t for t in trades if t['result']=='win']
losses = [t for t in trades if t['result']=='loss']
wr = len(wins)/total*100 if total else 0
avg_w = np.mean([t['pnl'] for t in wins]) if wins else 0
avg_l = np.mean([t['pnl'] for t in losses]) if losses else 0
pnl = sum(t['pnl'] for t in trades)

print(f"\n📊 OVERALL")
print(f"  Trades: {total} | Wins: {len(wins)} ({wr:.1f}%) | Losses: {len(losses)}")
print(f"  Avg win: +{avg_w:.1f} | Avg loss: {avg_l:.1f}")
print(f"  Total PnL: {pnl:+.1f} pts (${pnl*2:+.0f} MNQ)")
print(f"  Trades/day: {total/20:.1f} avg")

print(f"\n📍 BY SESSION")
sess = {}
for t in trades:
    s = t['session']
    if s not in sess: sess[s] = {'w':0,'l':0,'p':0}
    if t['result']=='win': sess[s]['w']+=1
    elif t['result']=='loss': sess[s]['l']+=1
    sess[s]['p']+=t['pnl']
for s,v in sorted(sess.items()):
    tot=v['w']+v['l']
    print(f"  {s:15s}: {tot:3d} trades | {v['w']/tot*100:.1f}% WR | {v['p']:+.0f} pts")

print(f"\n📈 BY DIRECTION")
for d in ['long','short']:
    dt=[t for t in trades if t['direction']==d]
    dw=[t for t in dt if t['result']=='win']
    if dt: print(f"  {d}: {len(dt)} trades | {len(dw)/len(dt)*100:.1f}% WR | {sum(t['pnl'] for t in dt):+.0f} pts")

print(f"\n📊 VOL RATIO ON WINS vs LOSSES")
wv=[t['vol_ratio'] for t in wins]
lv=[t['vol_ratio'] for t in losses]
if wv: print(f"  Wins avg vol ratio:   {sum(wv)/len(wv):.2f}x")
if lv: print(f"  Losses avg vol ratio: {sum(lv)/len(lv):.2f}x")

print(f"\n🎯 FVG vs OB ENTRY")
fvg_t=[t for t in trades if t['at_fvg']]
ob_t=[t for t in trades if t['at_ob'] and not t['at_fvg']]
fvg_w=[t for t in fvg_t if t['result']=='win']
ob_w=[t for t in ob_t if t['result']=='win']
if fvg_t: print(f"  FVG entries: {len(fvg_t)} | {len(fvg_w)/len(fvg_t)*100:.1f}% WR")
if ob_t:  print(f"  OB entries:  {len(ob_t)} | {len(ob_w)/len(ob_t)*100:.1f}% WR")

with open('backtest_v2_results.json','w') as f:
    json.dump(trades,f,indent=2)
print(f"\n💾 Saved to backtest_v2_results.json")
print(f"{'='*60}")
