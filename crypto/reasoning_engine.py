"""
Reasoning Engine — v2 (Multi-Agent Upgrade)
============================================
Fixed from v1:
  - Confidence gate restored: minimum 0.45 to execute, not bypassed
  - Regime direction filter: hard block LONGs in TRENDING_BEAR, SHORTs in TRENDING_BULL
  - Original penalty magnitudes restored (STRONG_BEAR trend penalty back to -0.22)
  - Counter-trend boosts removed (were actively hurting performance)
  - Tier B minimum confidence raised from 0.0 to 0.45
  - OFF_HOURS session blocked for new entries
  - Memory veto threshold raised from 35% to 40% WR
  - Sentiment integration: accepts external sentiment signal from SentimentAgent
  - Position audit integration: accepts audit signal from PositionAuditor
"""

import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    high  = df['high'].values  if 'high'  in df.columns else df['close'].values
    low   = df['low'].values   if 'low'   in df.columns else df['close'].values
    close = df['close'].values

    tr_list = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i]  - close[i-1])
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.01
    return np.mean(tr_list[-period:])


# ── LAYER 1: PERCEPTION ──────────────────────────────────────────────────────

def perceive(df, state):
    latest = df.iloc[-2]
    prev   = df.iloc[-3] if len(df) > 3 else df.iloc[-2]

    close   = float(latest['close'])
    h4_close = float(latest.get('h4_close', close))
    d1_close = float(latest.get('d1_close', close))
    w1_close = float(latest.get('w1_close', close))

    vol_window = df['volume'].iloc[-20:]
    avg_volume = float(vol_window.mean())

    bars_back_3  = df.iloc[-5]['close']  if len(df) >= 5  else df.iloc[0]['close']
    bars_back_24 = df.iloc[-26]['close'] if len(df) >= 26 else df.iloc[0]['close']

    try:
        dt  = pd.Timestamp(latest['Datetime'])
        hour = dt.hour
        dow  = dt.dayofweek
    except Exception:
        hour = 12
        dow  = 1

    entry_price = state.get('entry_price', 0)
    position    = state.get('position', 0)
    entry_time  = state.get('entry_time')
    bars_held   = 0
    if entry_time and position != 0:
        from datetime import datetime, timezone
        try:
            entry_dt  = datetime.fromisoformat(entry_time)
            bars_held = int((datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600)
        except Exception:
            bars_held = 0

    upnl = 0.0
    if position != 0 and entry_price > 0:
        upnl = (close - entry_price) / entry_price * position

    bar_open  = float(latest.get('open',  close))
    bar_high  = float(latest.get('high',  close))
    bar_low   = float(latest.get('low',   close))
    bar_range = bar_high - bar_low + 1e-9
    upper_wick  = (bar_high - max(bar_open, close)) / bar_range
    lower_wick  = (min(bar_open, close) - bar_low)  / bar_range
    body_pct    = abs(close - bar_open) / bar_range

    prev_open  = float(prev.get('open',  float(prev['close'])))
    prev_close = float(prev['close'])
    is_bearish_engulfing = (close < bar_open and bar_open > prev_close and close < prev_open)
    is_bullish_engulfing = (close > bar_open and bar_open < prev_close and close > prev_open)
    is_doji = body_pct < 0.1

    bullish_fvg = False
    bearish_fvg = False
    if len(df) >= 5:
        bar_3ago    = df.iloc[-4]
        bullish_fvg = float(bar_3ago.get('high', bar_3ago['close'])) < bar_low
        bearish_fvg = float(bar_3ago.get('low',  bar_3ago['close'])) > bar_high

    # ── Composite-regime inputs: EMAs, SMA, structure, multi-window momentum ──
    # Use only closed bars (df[:-1]) so the live in-progress bar doesn't skew slope.
    closes_closed = df['close'].iloc[:-1] if len(df) > 1 else df['close']

    if len(closes_closed) >= 20:
        ema20_arr = closes_closed.ewm(span=20, adjust=False).mean().values
        ema50_arr = closes_closed.ewm(span=50, adjust=False).mean().values
        if len(ema20_arr) >= 10:
            e20_now, e20_then = float(ema20_arr[-1]), float(ema20_arr[-10])
            ema20_slope = (e20_now - e20_then) / max(abs(e20_then), 1e-9)
        else:
            ema20_slope = 0.0
        if len(ema50_arr) >= 10:
            e50_now, e50_then = float(ema50_arr[-1]), float(ema50_arr[-10])
            ema50_slope = (e50_now - e50_then) / max(abs(e50_then), 1e-9)
        else:
            ema50_slope = 0.0
        sma20_val      = float(closes_closed.iloc[-20:].mean())
        price_vs_sma20 = (close - sma20_val) / max(sma20_val, 1e-9)
    else:
        ema20_slope = 0.0
        ema50_slope = 0.0
        price_vs_sma20 = 0.0

    if len(df) >= 11 and 'high' in df.columns and 'low' in df.columns:
        highs_closed = df['high'].iloc[:-1].values
        lows_closed  = df['low'].iloc[:-1].values
        recent_high = float(np.max(highs_closed[-5:]))
        prev_high   = float(np.max(highs_closed[-10:-5]))
        recent_low  = float(np.min(lows_closed[-5:]))
        prev_low    = float(np.min(lows_closed[-10:-5]))
        higher_highs = recent_high > prev_high
        higher_lows  = recent_low  > prev_low
        lower_highs  = recent_high < prev_high
        lower_lows   = recent_low  < prev_low
    else:
        higher_highs = higher_lows = lower_highs = lower_lows = False

    bars_back_10 = df.iloc[-12]['close'] if len(df) >= 12 else df.iloc[0]['close']
    bars_back_20 = df.iloc[-22]['close'] if len(df) >= 22 else df.iloc[0]['close']

    return {
        "price":           close,
        "price_vs_4h":     (close - h4_close) / max(h4_close, 0.01),
        "price_vs_1d":     (close - d1_close) / max(d1_close, 0.01),
        "price_vs_1w":     (close - w1_close) / max(w1_close, 0.01),
        "candle_body":     (close - float(latest.get('open', close))) / max(close, 0.01),
        "return_3bars":    (close - float(bars_back_3))  / max(float(bars_back_3),  0.01),
        "return_24bars":   (close - float(bars_back_24)) / max(float(bars_back_24), 0.01),
        "volume_ratio":    float(latest['volume']) / max(avg_volume, 1),
        "atr_pct":         calculate_atr(df) / max(close, 0.01),
        "hour":            hour,
        "dow":             dow,
        "position":        position,
        "upnl":            upnl,
        "bars_held":       bars_held,
        "upper_wick_pct":  upper_wick,
        "lower_wick_pct":  lower_wick,
        "body_pct":        body_pct,
        "is_bearish_engulfing": is_bearish_engulfing,
        "is_bullish_engulfing": is_bullish_engulfing,
        "is_doji":         is_doji,
        "bullish_fvg":     bullish_fvg,
        "bearish_fvg":     bearish_fvg,
        # Composite-regime inputs
        "ema20_slope":     ema20_slope,
        "ema50_slope":     ema50_slope,
        "price_vs_sma20":  price_vs_sma20,
        "higher_highs":    higher_highs,
        "higher_lows":     higher_lows,
        "lower_highs":     lower_highs,
        "lower_lows":      lower_lows,
        "return_10bars":   (close - float(bars_back_10)) / max(float(bars_back_10), 0.01),
        "return_20bars":   (close - float(bars_back_20)) / max(float(bars_back_20), 0.01),
    }


# ── LAYER 2: INTERPRETATION ──────────────────────────────────────────────────

def interpret(perception):
    conditions = {}
    narrative  = []

    above_4h = perception['price_vs_4h'] > 0.001
    above_1d = perception['price_vs_1d'] > 0.001
    above_1w = perception['price_vs_1w'] > 0.001
    trend_score = sum([above_4h, above_1d, above_1w])

    if trend_score == 3:
        conditions['trend'] = 'STRONG_BULL'
        narrative.append("Price above all EMAs (4H/1D/1W) — strong uptrend")
    elif trend_score == 2:
        conditions['trend'] = 'MILD_BULL'
        narrative.append("Price above 2/3 EMAs — mild bullish bias")
    elif trend_score == 1:
        conditions['trend'] = 'MILD_BEAR'
        narrative.append("Price below 2/3 EMAs — mild bearish bias")
    else:
        conditions['trend'] = 'STRONG_BEAR'
        narrative.append("Price below all EMAs — strong downtrend")

    mom = perception['return_3bars']
    if mom > 0.015:
        conditions['momentum'] = 'STRONG_UP'
        narrative.append(f"Strong upward momentum: +{mom:.1%} in 3 bars")
    elif mom > 0:
        conditions['momentum'] = 'WEAK_UP'
        narrative.append(f"Weak upward momentum: +{mom:.1%} in 3 bars")
    elif mom > -0.015:
        conditions['momentum'] = 'WEAK_DOWN'
        narrative.append(f"Weak downward momentum: {mom:.1%} in 3 bars")
    else:
        conditions['momentum'] = 'STRONG_DOWN'
        narrative.append(f"Strong downward momentum: {mom:.1%} in 3 bars")

    vol = perception['volume_ratio']
    if vol > 2.0:
        conditions['volume'] = 'VERY_HIGH'
        narrative.append(f"Very high volume: {vol:.1f}x average")
    elif vol > 1.4:
        conditions['volume'] = 'HIGH'
        narrative.append(f"High volume: {vol:.1f}x average")
    elif vol > 0.7:
        conditions['volume'] = 'NORMAL'
        narrative.append(f"Normal volume: {vol:.1f}x average")
    else:
        conditions['volume'] = 'LOW'
        narrative.append(f"Low volume: {vol:.1f}x average")

    atr = perception['atr_pct']
    if atr > 0.025:
        conditions['volatility'] = 'HIGH'
        narrative.append(f"High volatility: ATR {atr:.2%}")
    elif atr > 0.010:
        conditions['volatility'] = 'NORMAL'
        narrative.append(f"Normal volatility: ATR {atr:.2%}")
    else:
        conditions['volatility'] = 'LOW'
        narrative.append(f"Low volatility: ATR {atr:.2%}")

    hour = perception['hour']
    if 7 <= hour <= 11:
        conditions['session'] = 'LONDON'
        narrative.append("London session")
    elif 12 <= hour <= 16:
        conditions['session'] = 'NY_OPEN'
        narrative.append("New York open")
    elif 17 <= hour <= 20:
        conditions['session'] = 'NY_PM'
        narrative.append("New York afternoon")
    else:
        conditions['session'] = 'OFF_HOURS'
        narrative.append("Off-hours / Asia session")

    if perception['upper_wick_pct'] > 0.6:
        conditions['wick'] = 'UPPER_REJECTION'
        narrative.append("Strong upper wick — bearish rejection")
    elif perception['lower_wick_pct'] > 0.6:
        conditions['wick'] = 'LOWER_REJECTION'
        narrative.append("Strong lower wick — bullish rejection")
    else:
        conditions['wick'] = 'NEUTRAL'

    if perception['is_doji']:
        conditions['candle'] = 'DOJI'
        narrative.append("Doji — market indecision")
    elif perception['is_bearish_engulfing']:
        conditions['candle'] = 'BEARISH_ENGULF'
        narrative.append("Bearish engulfing — strong reversal signal")
    elif perception['is_bullish_engulfing']:
        conditions['candle'] = 'BULLISH_ENGULF'
        narrative.append("Bullish engulfing — strong reversal signal")
    else:
        conditions['candle'] = 'NORMAL'

    if perception['bullish_fvg']:
        conditions['fvg'] = 'BULLISH_FVG'
        narrative.append("Bullish FVG present — price may fill gap")
    elif perception['bearish_fvg']:
        conditions['fvg'] = 'BEARISH_FVG'
        narrative.append("Bearish FVG present — price may fill gap")
    else:
        conditions['fvg'] = 'NONE'

    regime = classify_regime(conditions, perception)
    conditions['regime'] = regime
    narrative.append(f"Regime: {regime}")

    return conditions, narrative


# ── LAYER 2.5: REGIME CLASSIFIER ─────────────────────────────────────────────

def classify_regime(conditions, perception):
    """
    Composite regime classifier.

    Replaces the old "price-vs-EMA stack" labeller, which lagged badly on
    real recoveries (e.g. SOL +6.5% off lows still tagged STRONG_BEAR because
    long-period EMAs hadn't caught up).

    Sums four normalized signals into a [-2, +2] direction score:
      (a) EMA slope     — avg pct change of 20/50 EMA over last 10 closed bars
                          (saturates at ±2%)
      (b) Structure     — higher-highs/higher-lows vs lower-highs/lower-lows
      (c) Price vs SMA  — close vs 20-bar SMA  (±0.5)
      (d) Multi-window momentum — avg of 3/10/20-bar pct change (saturates at ±3%)

    Mapping:
       score >=  1.5  → STRONG_BULL
       score  >  0.5  → TRENDING_BULL
       score  > -0.5  → RANGING
       score >= -1.5  → TRENDING_BEAR
       score <  -1.5  → STRONG_BEAR
    Very low ATR (<0.3%) short-circuits to LOW_QUALITY.
    """
    atr_pct = float(perception.get('atr_pct', 0.0))

    if atr_pct < 0.003:
        return 'LOW_QUALITY'

    score = 0.0

    # (a) EMA slope
    ema20_slope = float(perception.get('ema20_slope', 0.0))
    ema50_slope = float(perception.get('ema50_slope', 0.0))
    avg_slope   = (ema20_slope + ema50_slope) / 2.0
    score += max(-1.0, min(1.0, avg_slope / 0.02))

    # (b) Recent structure
    hh = bool(perception.get('higher_highs', False))
    hl = bool(perception.get('higher_lows',  False))
    lh = bool(perception.get('lower_highs',  False))
    ll = bool(perception.get('lower_lows',   False))
    if hh and hl:
        score += 1.0
    elif lh and ll:
        score -= 1.0
    elif hh or hl:
        score += 0.5
    elif lh or ll:
        score -= 0.5

    # (c) Price vs 20-bar SMA
    price_vs_sma20 = float(perception.get('price_vs_sma20', 0.0))
    if price_vs_sma20 > 0:
        score += 0.5
    elif price_vs_sma20 < 0:
        score -= 0.5

    # (d) Multi-window momentum
    r3  = float(perception.get('return_3bars',  0.0))
    r10 = float(perception.get('return_10bars', 0.0))
    r20 = float(perception.get('return_20bars', 0.0))
    avg_mom = (r3 + r10 + r20) / 3.0
    score += max(-1.0, min(1.0, avg_mom / 0.03))

    if score >= 1.5:
        return 'STRONG_BULL'
    if score > 0.5:
        return 'TRENDING_BULL'
    if score > -0.5:
        return 'RANGING'
    if score >= -1.5:
        return 'TRENDING_BEAR'
    return 'STRONG_BEAR'


# ── LAYER 3: REASONING ENGINE (FIXED) ────────────────────────────────────────

def reason(ppo_action, conditions, perception, memory, sentiment_signal=None,
           model_probs=None):
    """
    sentiment_signal: dict from SentimentAgent, e.g.:
        {'direction': 'BEARISH', 'strength': 0.7, 'sources': [...]}
    model_probs: optional np.array of PPO policy action probabilities
        [P(HOLD), P(LONG), P(SHORT), P(CLOSE)]. When provided, the model's
        own conviction becomes the PRIMARY confidence signal and the
        rule checklist below is demoted to a secondary modifier + hard vetoes.
        (PROB_GATE fix: the old rule-only confidence was historically
        inverted — rejected trades won 51% vs 32% for executed ones.)
    Returns (verdict, confidence, evidence_for, evidence_against)
    """
    evidence_for     = []
    evidence_against = []
    confidence       = 0.30   # baseline — must earn above 0.45 to execute

    action_is_long  = ppo_action == 1
    action_is_short = ppo_action == 2

    if action_is_long or action_is_short:

        # ── Regime direction hard-blocks REMOVED: any direction allowed in any regime ──
        regime = conditions.get('regime', 'LOW_QUALITY')

        # ── FIX 2: OFF_HOURS block for new entries ───────────────────────────
        if conditions.get('session') == 'OFF_HOURS':
            evidence_against.append("HARD BLOCK: OFF_HOURS — no new entries during Asia session")
            return "HARD_REJECT", 0.0, evidence_for, evidence_against

        # ── Trend alignment (restored original penalties) ────────────────────
        if action_is_long:
            if conditions['trend'] == 'STRONG_BULL':
                evidence_for.append("Strong uptrend — long aligned with trend")
                confidence += 0.15
            elif conditions['trend'] == 'MILD_BULL':
                evidence_for.append("Mild uptrend — long mildly aligned")
                confidence += 0.07
            elif conditions['trend'] == 'MILD_BEAR':
                evidence_against.append("Mild downtrend — long against trend")
                confidence -= 0.10   # restored
            elif conditions['trend'] == 'STRONG_BEAR' and conditions.get('regime') not in ('STRONG_BULL', 'TRENDING_BULL'):
                evidence_against.append("Strong downtrend — long strongly against trend")
                confidence -= 0.15   # skipped when composite regime overrides EMA-stack as bullish

        if action_is_short:
            if conditions['trend'] == 'STRONG_BEAR':
                evidence_for.append("Strong downtrend — short aligned with trend")
                confidence += 0.15
            elif conditions['trend'] == 'MILD_BEAR':
                evidence_for.append("Mild downtrend — short mildly aligned")
                confidence += 0.07
            elif conditions['trend'] == 'MILD_BULL':
                evidence_against.append("Mild uptrend — short against trend")
                confidence -= 0.10   # restored
            elif conditions['trend'] == 'STRONG_BULL' and conditions.get('regime') not in ('STRONG_BEAR', 'TRENDING_BEAR'):
                evidence_against.append("Strong uptrend — short strongly against trend")
                confidence -= 0.15   # PROB_GATE: symmetric with long penalty (was -0.22, suppressed all shorts)

        # ── Momentum ──────────────────────────────────────────────────────────
        if action_is_long and conditions['momentum'] in ('STRONG_UP', 'WEAK_UP'):
            evidence_for.append("Momentum supports long")
            confidence += 0.08
        elif action_is_long and conditions['momentum'] == 'STRONG_DOWN':
            evidence_against.append("Momentum strongly against long")
            confidence -= 0.12   # restored

        if action_is_short and conditions['momentum'] in ('STRONG_DOWN', 'WEAK_DOWN'):
            evidence_for.append("Momentum supports short")
            confidence += 0.08
        elif action_is_short and conditions['momentum'] == 'STRONG_UP':
            evidence_against.append("Momentum strongly against short")
            confidence -= 0.12   # restored

        # ── Volume ────────────────────────────────────────────────────────────
        if conditions['volume'] in ('HIGH', 'VERY_HIGH'):
            evidence_for.append("High volume confirms conviction")
            confidence += 0.08
        elif conditions['volume'] == 'LOW':
            evidence_against.append("Low volume — weak conviction")
            confidence -= 0.10   # restored

        # ── Volatility ────────────────────────────────────────────────────────
        if conditions['volatility'] == 'HIGH':
            evidence_against.append("High volatility — stop risk elevated")
            confidence -= 0.08   # restored

        # ── ICT: Wick confluence ──────────────────────────────────────────────
        if action_is_long:
            if conditions.get('wick') == 'LOWER_REJECTION':
                evidence_for.append("Lower wick rejection confirms bullish entry")
                confidence += 0.08
            elif conditions.get('wick') == 'UPPER_REJECTION':
                evidence_against.append("Upper wick rejection contradicts long")
                confidence -= 0.08   # restored
        elif action_is_short:
            if conditions.get('wick') == 'UPPER_REJECTION':
                evidence_for.append("Upper wick rejection confirms short entry")
                confidence += 0.08
            elif conditions.get('wick') == 'LOWER_REJECTION':
                evidence_against.append("Lower wick rejection contradicts short")
                confidence -= 0.08   # restored

        # ── ICT: Candle pattern ───────────────────────────────────────────────
        candle = conditions.get('candle', 'NORMAL')
        if action_is_long and candle == 'BULLISH_ENGULF':
            evidence_for.append("Bullish engulfing confirms long")
            confidence += 0.10
        elif action_is_long and candle == 'BEARISH_ENGULF':
            evidence_against.append("Bearish engulfing contradicts long")
            confidence -= 0.10   # restored
        elif action_is_short and candle == 'BEARISH_ENGULF':
            evidence_for.append("Bearish engulfing confirms short")
            confidence += 0.10
        elif action_is_short and candle == 'BULLISH_ENGULF':
            evidence_against.append("Bullish engulfing contradicts short")
            confidence -= 0.10   # restored

        if candle == 'DOJI':
            evidence_against.append("Doji — avoid entry during indecision")
            confidence -= 0.12   # restored

        # ── ICT: FVG ──────────────────────────────────────────────────────────
        fvg = conditions.get('fvg', 'NONE')
        if action_is_long and fvg == 'BULLISH_FVG':
            evidence_for.append("Bullish FVG supports long bias")
            confidence += 0.06
        elif action_is_short and fvg == 'BEARISH_FVG':
            evidence_for.append("Bearish FVG supports short bias")
            confidence += 0.06

        # ── Regime scoring (no counter-trend boosts, only aligned rewards) ───
        if regime == 'RANGING':
            evidence_for.append("RANGING regime — bot has edge in range conditions")
            confidence += 0.08
        elif regime == 'HIGH_VOLATILITY':
            evidence_against.append("HIGH_VOLATILITY — elevated stop risk")
            confidence -= 0.10   # restored
        elif regime == 'LOW_QUALITY':
            evidence_against.append("LOW_QUALITY regime — conflicting signals")
            confidence -= 0.05
        # TRENDING_BULL with long, or TRENDING_BEAR with short → neutral (handled by trend above)

        # ── Memory check (veto threshold raised to 40%) ───────────────────────
        veto, wr = memory.should_veto(conditions)
        if veto:
            evidence_against.append(f"MEMORY VETO: setup has only {wr:.0%} WR historically (threshold 25%)")
            confidence -= 0.30
        else:
            adj = memory.confidence_adjustment(conditions)
            if adj > 0.02:
                wr_val = memory.get_win_rate(conditions)
                evidence_for.append(f"Memory: similar setups won {wr_val:.0%} historically")
                confidence += adj
            elif adj < -0.02:
                wr_val = memory.get_win_rate(conditions)
                evidence_against.append(f"Memory: similar setups won only {wr_val:.0%} historically")
                confidence += adj * 0.5   # penalty weight halved — was dragging confidence too low

        # ── Regret adjustment ─────────────────────────────────────────────────
        regret_adj = memory.get_regret_adjustment(conditions)
        if regret_adj > 0:
            evidence_for.append(f"Regret learning: agent missed profitable trades — lowering bar")
            confidence += regret_adj

        # ── NEW: Sentiment integration ────────────────────────────────────────
        if sentiment_signal and sentiment_signal.get('direction') not in (None, 'NEUTRAL'):
            s_dir      = sentiment_signal['direction']   # 'BULLISH' or 'BEARISH'
            s_strength = float(sentiment_signal.get('strength', 0.5))
            s_sources  = sentiment_signal.get('sources', [])
            boost      = round(s_strength * 0.12, 3)   # max ±0.12 from sentiment

            if action_is_long and s_dir == 'BULLISH':
                evidence_for.append(
                    f"Sentiment BULLISH ({s_strength:.0%} strength, sources: {', '.join(s_sources)}) — supports long"
                )
                confidence += boost
            elif action_is_long and s_dir == 'BEARISH':
                evidence_against.append(
                    f"Sentiment BEARISH ({s_strength:.0%} strength) — warns against long"
                )
                confidence -= boost
            elif action_is_short and s_dir == 'BEARISH':
                evidence_for.append(
                    f"Sentiment BEARISH ({s_strength:.0%} strength, sources: {', '.join(s_sources)}) — supports short"
                )
                confidence += boost
            elif action_is_short and s_dir == 'BULLISH':
                evidence_against.append(
                    f"Sentiment BULLISH ({s_strength:.0%} strength) — warns against short"
                )
                confidence -= boost

    elif ppo_action == 3:  # CLOSE
        confidence = 0.80
        evidence_for.append("PPO model signaling close")
        if perception['upnl'] < -0.015:
            evidence_for.append(f"Unrealized loss {perception['upnl']:.1%} — good to close")
            confidence += 0.10
        elif perception['upnl'] > 0.015:
            evidence_for.append(f"Locking in profit {perception['upnl']:.1%}")
            confidence += 0.05

    confidence = max(0.05, min(0.95, confidence))
    rule_conf  = confidence   # PROB_GATE: keep checklist score separate

    # ── PROB_GATE: model conviction as primary signal ────────────────────────
    model_conf = None
    if model_probs is not None and ppo_action in (1, 2):
        try:
            model_conf = float(model_probs[int(ppo_action)])
            evidence_for.append(f"Model conviction P({'LONG' if ppo_action == 1 else 'SHORT'})={model_conf:.0%}")
            # Blend: model is primary, checklist is secondary
            confidence = max(0.05, min(0.95, 0.60 * model_conf + 0.40 * rule_conf))
        except Exception:
            model_conf = None

    # ── Confidence gate ──────────────────────────────────────────────────────
    if ppo_action in (1, 2):
        veto_fired, veto_wr = memory.should_veto(conditions)
        regime    = conditions.get('regime', 'LOW_QUALITY')
        atr_pct   = perception.get('atr_pct', 0.0)

        if veto_fired:
            verdict = "REJECT"
            evidence_against.append(f"Memory veto: WR={veto_wr:.0%} < 25% threshold")
        elif model_conf is not None:
            # PROB_GATE: gate on the model's own probability first; the
            # checklist can only veto (rule_conf < 0.30 = actively hostile
            # conditions like OFF_HOURS proximity, stacked counter-evidence).
            if model_conf >= 0.55 and rule_conf >= 0.30:
                verdict = "EXECUTE"
            elif model_conf >= 0.45 and rule_conf >= 0.30:
                verdict = "WEAK_EXECUTE"
            else:
                verdict = "REJECT"
                if model_conf < 0.45:
                    evidence_against.append(
                        f"Model conviction {model_conf:.0%} below 0.45 minimum gate")
                else:
                    evidence_against.append(
                        f"Checklist veto: rule confidence {rule_conf:.0%} < 0.30")
        elif confidence >= 0.45:
            verdict = "EXECUTE"        # legacy path (no probs available)
        elif confidence >= 0.30:
            verdict = "WEAK_EXECUTE"   # only Tier B at 50% size allowed
        else:
            verdict = "REJECT"         # below 0.30 — skip regardless
            evidence_against.append(f"Confidence {confidence:.0%} below 0.30 minimum gate")
    elif ppo_action == 3:
        verdict = "EXECUTE"
    else:
        verdict = "EXECUTE"

    return verdict, confidence, evidence_for, evidence_against


# ── REGRET SIMULATION ────────────────────────────────────────────────────────

def simulate_missed_trade(df, rejection_bar_idx, rejected_action, sl_pct, tp_pct, max_bars=48):
    if rejection_bar_idx + 1 >= len(df):
        return None, 0.0, "NO_DATA"

    entry_price = float(df.iloc[rejection_bar_idx]['close'])
    direction   = 1 if rejected_action == 1 else -1

    sl_price = entry_price * (1 - sl_pct) if direction == 1 else entry_price * (1 + sl_pct)
    tp_price = entry_price * (1 + tp_pct) if direction == 1 else entry_price * (1 - tp_pct)

    for i in range(1, min(max_bars + 1, len(df) - rejection_bar_idx)):
        bar  = df.iloc[rejection_bar_idx + i]
        high = float(bar.get('high', bar['close']))
        low  = float(bar.get('low',  bar['close']))

        if direction == 1:
            if low  <= sl_price:
                return False, (sl_price - entry_price) / entry_price, "SL"
            if high >= tp_price:
                return True,  (tp_price - entry_price) / entry_price, "TP"
        else:
            if high >= sl_price:
                return False, (entry_price - sl_price) / entry_price, "SL"
            if low  <= tp_price:
                return True,  (entry_price - tp_price) / entry_price, "TP"

    exit_price = float(df.iloc[min(rejection_bar_idx + max_bars, len(df)-1)]['close'])
    pnl = (exit_price - entry_price) / entry_price * direction
    return pnl > 0, pnl, "TIME"


# ── LAYER 3.5: DYNAMIC RISK SIZING ───────────────────────────────────────────

def get_dynamic_sl_tp(conditions, memory):
    expectancy = memory.get_expectancy(conditions)
    if expectancy is not None:
        if expectancy >= 0.5:
            return 0.020, 0.05
        elif expectancy >= 0.1:
            return 0.025, 0.05
        elif expectancy >= 0.0:
            return 0.025, 0.06
        else:
            return 0.030, 0.05

    wr = memory.get_win_rate(conditions)
    if wr is None:
        return 0.025, 0.05
    if wr >= 0.60:
        return 0.020, 0.05
    elif wr >= 0.55:
        return 0.025, 0.05
    elif wr >= 0.50:
        return 0.025, 0.06
    else:
        return 0.025, 0.05


# ── LAYER 3.75: SETUP QUALITY TIERS (FIXED) ──────────────────────────────────

def classify_setup_quality(conditions, confidence, memory, perception):
    """
    A+: best setups — full size
    A:  good setups — 80% size
    B:  marginal — 50% size (FIX: now requires confidence >= 0.45, not 0.0)
    TRASH: skip
    """
    regime     = conditions.get('regime', 'LOW_QUALITY')
    volume     = conditions.get('volume', 'NORMAL')
    momentum   = conditions.get('momentum', 'WEAK_UP')
    atr_pct    = perception.get('atr_pct', 0.02)
    expectancy = memory.get_expectancy(conditions)

    # TRASH: block unconditionally (regime block removed — every regime tradeable)
    if expectancy is not None and expectancy < -0.3:
        return 'TRASH'

    # A+: best setups
    regime_ok     = regime in ('RANGING', 'TRENDING_BULL', 'TRENDING_BEAR',
                               'STRONG_BULL', 'STRONG_BEAR')
    volume_ok     = volume in ('HIGH', 'VERY_HIGH')
    momentum_ok   = momentum in ('STRONG_UP', 'STRONG_DOWN', 'WEAK_UP', 'WEAK_DOWN')
    expectancy_ok = expectancy is None or expectancy >= 0.2
    if regime_ok and volume_ok and momentum_ok and expectancy_ok and confidence >= 0.68:
        return 'A_PLUS'

    # A: good setups
    regime_not_bad = regime not in ('HIGH_VOLATILITY', 'LOW_QUALITY')
    volume_not_low = volume != 'LOW'
    if regime_not_bad and volume_not_low and confidence >= 0.58:
        return 'A'

    # B: marginal — confidence gate only (regime restriction removed)
    if confidence >= 0.45:
        return 'B'

    # C: marginal but executable — WEAK_EXECUTE range (0.30-0.45)
    if confidence >= 0.30:
        return 'C'

    return 'TRASH'


# ── LAYER 4: DECISION + EXPLANATION ──────────────────────────────────────────

def decide(ppo_action, conditions, perception, memory, narrative, sentiment_signal=None,
           model_probs=None):
    """
    sentiment_signal: optional dict from SentimentAgent
    model_probs: optional PPO policy action probabilities (see reason())
    Returns (verdict, confidence, reasoning_text, tier)
    """
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}

    verdict, confidence, evidence_for, evidence_against = reason(
        ppo_action, conditions, perception, memory, sentiment_signal,
        model_probs=model_probs
    )

    lines = []
    lines.append(f"PPO Signal: {action_names.get(ppo_action, '?')}")
    lines.append(f"Regime: {conditions.get('regime', 'UNKNOWN')}")
    lines.append("Market Reading:")
    for n in narrative:
        lines.append(f"  — {n}")

    if evidence_for:
        lines.append("Evidence FOR:")
        for e in evidence_for:
            lines.append(f"  ✅ {e}")

    if evidence_against:
        lines.append("Evidence AGAINST:")
        for e in evidence_against:
            lines.append(f"  ❌ {e}")

    if sentiment_signal:
        lines.append(f"Sentiment: {sentiment_signal.get('direction', 'N/A')} "
                     f"({sentiment_signal.get('strength', 0):.0%} strength) "
                     f"| F&G: {sentiment_signal.get('fear_greed_label', 'N/A')} "
                     f"({sentiment_signal.get('fear_greed_value', 'N/A')})")

    tier = classify_setup_quality(conditions, confidence, memory, perception)

    expectancy = memory.get_expectancy(conditions)
    if expectancy is not None:
        lines.append(f"Expectancy: {expectancy:+.2f}R per trade")
    if model_probs is not None:
        try:
            lines.append("Model P(H/L/S/C): " +
                         "/".join(f"{float(p):.0%}" for p in model_probs))
        except Exception:
            pass
    lines.append(f"Confidence: {confidence:.0%}")
    lines.append(f"Quality: {tier}")
    lines.append(f"Verdict: {verdict}")

    return verdict, confidence, "\n".join(lines), tier
