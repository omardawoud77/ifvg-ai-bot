import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("📥 Downloading 30 days of NQ 5m data...")
df = yf.download("NQ=F", period="30d", interval="5m", progress=False)
df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
df = df.dropna()
print(f"✅ Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# ── ICT Scoring (same as live bot) ─────────────────────────────────────
def get_mtf_bias(df_full, idx):
    """Get weekly/daily/4h bias at a given bar index."""
    try:
        slice_df = df_full.iloc[max(0,idx-500):idx+1]
        def ema(s, n): return s.ewm(span=n).mean()
        close = slice_df['close']
        # Weekly bias: EMA 50 vs 200 on daily-equivalent (288 5m bars = 1 day)
        weekly = 'bullish' if ema(close,576).iloc[-1] > ema(close,1440).iloc[-1] else 'bearish'
        daily  = 'bullish' if ema(close,288).iloc[-1] > ema(close,576).iloc[-1] else 'bearish'
        h4     = 'bullish' if ema(close,48).iloc[-1]  > ema(close,96).iloc[-1]  else 'bearish'
        h1     = 'bullish' if ema(close,12).iloc[-1]  > ema(close,24).iloc[-1]  else 'bearish'
        return {'weekly': weekly, 'daily': daily, '4h': h4, '1h': h1}
    except:
        return {'weekly': '', 'daily': '', '4h': '', '1h': ''}

def ict_score(df_slice, direction, mtf, kz_active):
    conds = {}
    if len(df_slice) < 20:
        return 0, conds
    last = df_slice.iloc[-1]
    close = float(last['close'])
    volume = float(last.get('volume', 0))
    avg_vol = float(df_slice['volume'].tail(20).mean()) if 'volume' in df_slice.columns else 0

    # 1. HTF aligned
    weekly, daily, h4 = mtf.get('weekly',''), mtf.get('daily',''), mtf.get('4h','')
    if direction == 'long':
        htf_ok = (weekly == 'bullish' and daily == 'bullish' and h4 == 'bullish')
    else:
        htf_ok = (weekly == 'bearish' and daily == 'bearish' and h4 == 'bearish')
    conds['htf_aligned'] = htf_ok

    # 2. Kill zone
    conds['kill_zone'] = kz_active

    # 3. FVG
    fvg_hit = False
    for i in range(2, min(15, len(df_slice))):
        c1 = df_slice.iloc[-i-1]
        c3 = df_slice.iloc[-i+1] if i > 1 else last
        gap_low = float(c1['high']); gap_high = float(c3['low'])
        if gap_high > gap_low and direction == 'long' and gap_low <= close <= gap_high:
            fvg_hit = True; break
        gap_low2 = float(c3['high']); gap_high2 = float(c1['low'])
        if gap_high2 < gap_low2 and direction == 'short' and gap_high2 <= close <= gap_low2:
            fvg_hit = True; break
    conds['fvg_hit'] = fvg_hit

    # 4. Order Block
    ob_hit = False
    avg_range = float((df_slice.tail(20)['high'] - df_slice.tail(20)['low']).mean())
    for i in range(3, min(20, len(df_slice))):
        bar = df_slice.iloc[-i]
        if float(bar['high']) - float(bar['low']) > avg_range * 1.5:
            if float(bar['low']) <= close <= float(bar['high']):
                ob_hit = True; break
    conds['ob_hit'] = ob_hit

    # 5. BOS
    swing_highs = df_slice.tail(10)['high']
    swing_lows  = df_slice.tail(10)['low']
    if direction == 'long':
        conds['bos'] = close > float(swing_highs.iloc[-5:-1].max())
    else:
        conds['bos'] = close < float(swing_lows.iloc[-5:-1].min())

    # 6. Volume
    conds['volume_ok'] = (avg_vol > 0 and volume > avg_vol * 0.8)

    # 7. Momentum alignment — EMA 9 vs EMA 21 on 5m confirms direction
    ema9  = float(df_slice['close'].ewm(span=9).mean().iloc[-1])
    ema21 = float(df_slice['close'].ewm(span=21).mean().iloc[-1])
    if direction == 'long':
        conds['momentum_ok'] = ema9 > ema21
    else:
        conds['momentum_ok'] = ema9 < ema21

    # 8. RSI condition — buy oversold, sell overbought
    if len(df_slice) >= 15:
        delta = df_slice['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = float((100 - 100/(1+rs)).iloc[-1])
        if direction == 'long':
            conds['rsi_ok'] = rsi < 55  # below midpoint
        else:
            conds['rsi_ok'] = rsi > 45  # above midpoint
    else:
        conds['rsi_ok'] = False

    return sum(1 for v in conds.values() if v), conds  # now 8 conditions max

def get_session(dt):
    """Get session name from UTC datetime."""
    h = dt.hour * 60 + dt.minute  # minutes since midnight UTC
    if 0 <= h < 360:   return "asia", False  # TEST: block Asia
    if 360 <= h < 480: return "london_open", True
    if 480 <= h < 720: return "london", True
    if 720 <= h < 780: return "transition_london_ny", False  # BLOCKED
    if 780 <= h < 960: return "ny_open", True
    if 960 <= h < 1080: return "ny_pm", True
    if 1080 <= h < 1200: return "ny_close", True
    return "transition", False  # BLOCKED

# ── Backtest Engine ─────────────────────────────────────────────────────
RR = 2.0
THRESHOLD = 5  # out of 8 now
MAX_DAILY = 5
COOLDOWN_BARS = 6  # 30 min = 6 x 5m bars

trades = []
last_trade_bar = -999
daily_counts = {}

print("\n🔄 Running backtest...")

for i in range(200, len(df)):
    bar = df.iloc[i]
    bar_time = df.index[i]
    
    # Skip if already in trade (simplified - just check cooldown)
    if i - last_trade_bar < COOLDOWN_BARS:
        continue
    
    # Session check
    session, kz_active = get_session(bar_time)
    if not kz_active:
        continue
    
    # Daily limit
    day_key = bar_time.strftime("%Y-%m-%d")
    if daily_counts.get(day_key, 0) >= MAX_DAILY:
        continue
    
    # Get MTF bias
    mtf = get_mtf_bias(df, i)
    
    # Determine direction from HTF
    h4 = mtf.get('4h', '')
    if not h4:
        continue
    direction = 'long' if h4 == 'bullish' else 'short'
    
    # ICT score
    df_slice = df.iloc[max(0,i-50):i+1]
    score, conds = ict_score(df_slice, direction, mtf, kz_active)
    
    # Quality filters
    if score < THRESHOLD:
        continue
    if not conds.get('htf_aligned', False):
        continue
    if not conds.get('volume_ok', False):
        continue
    
    # Entry
    entry = float(bar['close'])
    atr = float((df.iloc[max(0,i-20):i]['high'] - df.iloc[max(0,i-20):i]['low']).mean())
    sl_dist = max(25.0, min(60.0, atr * 1.0))  # wider SL, ATR x1.0
    
    if direction == 'long':
        sl = entry - sl_dist
        tp = entry + sl_dist * RR
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * RR
    
    # NY open uses 1RR, all others use 2RR
    rr_multiplier = 1.0 if session == 'ny_open' else RR
    if direction == 'long':
        tp = entry + sl_dist * rr_multiplier
    else:
        tp = entry - sl_dist * rr_multiplier

    # Simulate outcome on next bars
    result = None
    exit_price = None
    for j in range(i+1, min(i+100, len(df))):
        future = df.iloc[j]
        if direction == 'long':
            if float(future['low']) <= sl:
                result = 'loss'; exit_price = sl; break
            if float(future['high']) >= tp:
                result = 'win'; exit_price = tp; break
        else:
            if float(future['high']) >= sl:
                result = 'loss'; exit_price = sl; break
            if float(future['low']) <= tp:
                result = 'win'; exit_price = tp; break
    
    if not result:
        result = 'timeout'; exit_price = float(df.iloc[min(i+99, len(df)-1)]['close'])
    
    pnl_pts = (exit_price - entry) if direction == 'long' else (entry - exit_price)
    
    trades.append({
        'time': str(bar_time),
        'session': session,
        'direction': direction,
        'entry': round(entry, 2),
        'sl': round(sl, 2),
        'tp': round(tp, 2),
        'exit_price': round(exit_price, 2),
        'pnl_pts': round(pnl_pts, 1),
        'result': result,
        'score': score,
        'conds': conds,
        'htf': mtf,
        'sl_dist': round(sl_dist, 1)
    })
    
    last_trade_bar = i
    daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

# ── Analysis ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BACKTEST RESULTS — Last 30 Days")
print(f"{'='*60}")

total = len(trades)
wins = [t for t in trades if t['result'] == 'win']
losses = [t for t in trades if t['result'] == 'loss']
timeouts = [t for t in trades if t['result'] == 'timeout']

wr = len(wins)/total*100 if total > 0 else 0
avg_win = np.mean([t['pnl_pts'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl_pts'] for t in losses]) if losses else 0
total_pnl = sum(t['pnl_pts'] for t in trades)

print(f"\n📊 OVERALL")
print(f"  Total trades : {total}")
print(f"  Wins         : {len(wins)} ({wr:.1f}%)")
print(f"  Losses       : {len(losses)}")
print(f"  Timeouts     : {len(timeouts)}")
print(f"  Avg win      : +{avg_win:.1f} pts")
print(f"  Avg loss     : {avg_loss:.1f} pts")
print(f"  Total PnL    : {total_pnl:+.1f} pts (${total_pnl*2:+.0f} MNQ)")

# By session
print(f"\n📍 BY SESSION")
sessions = {}
for t in trades:
    s = t['session']
    if s not in sessions: sessions[s] = {'w':0,'l':0,'pnl':0}
    if t['result'] == 'win': sessions[s]['w'] += 1
    elif t['result'] == 'loss': sessions[s]['l'] += 1
    sessions[s]['pnl'] += t['pnl_pts']
for s, v in sorted(sessions.items()):
    tot = v['w']+v['l']
    wr_s = v['w']/tot*100 if tot > 0 else 0
    print(f"  {s:20s}: {tot:3d} trades | {wr_s:5.1f}% WR | {v['pnl']:+.0f} pts")

# By direction
print(f"\n📈 BY DIRECTION")
for d in ['long', 'short']:
    dt = [t for t in trades if t['direction'] == d]
    dw = [t for t in dt if t['result'] == 'win']
    wr_d = len(dw)/len(dt)*100 if dt else 0
    pnl_d = sum(t['pnl_pts'] for t in dt)
    print(f"  {d:6s}: {len(dt):3d} trades | {wr_d:5.1f}% WR | {pnl_d:+.0f} pts")

# By ICT score
print(f"\n🎯 BY ICT SCORE")
for sc in range(4, 8):
    st = [t for t in trades if t['score'] == sc]
    sw = [t for t in st if t['result'] == 'win']
    wr_sc = len(sw)/len(st)*100 if st else 0
    pnl_sc = sum(t['pnl_pts'] for t in st)
    print(f"  Score {sc}/7: {len(st):3d} trades | {wr_sc:5.1f}% WR | {pnl_sc:+.0f} pts")

# Condition win rates
print(f"\n⚡ CONDITION WIN RATES (when condition is True)")
cond_names = ['htf_aligned','kill_zone','fvg_hit','ob_hit','bos','volume_ok','momentum_ok','rsi_ok']
for cn in cond_names:
    with_cond  = [t for t in trades if t['conds'].get(cn)]
    w_with = [t for t in with_cond if t['result'] == 'win']
    wr_c = len(w_with)/len(with_cond)*100 if with_cond else 0
    print(f"  {cn:20s}: {len(with_cond):3d} trades | {wr_c:5.1f}% WR")

# Save results
with open('backtest_results.json', 'w') as f:
    json.dump(trades, f, indent=2)
print(f"\n💾 Full results saved to backtest_results.json")
print(f"{'='*60}")
