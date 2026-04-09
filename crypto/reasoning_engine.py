"""
Reasoning Engine
================
5-layer system that reads the market like a professional trader.
No external APIs. Pure code and logic.
"""

import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    high = df['high'].values if 'high' in df.columns else df['close'].values
    low = df['low'].values if 'low' in df.columns else df['close'].values
    close = df['close'].values

    tr_list = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.01
    return np.mean(tr_list[-period:])


# ── LAYER 1: PERCEPTION ──────────────────────────────────────────────────────

def perceive(df, state):
    latest = df.iloc[-2]
    prev = df.iloc[-3] if len(df) > 3 else df.iloc[-2]

    close = float(latest['close'])
    h4_close = float(latest.get('h4_close', close))
    d1_close = float(latest.get('d1_close', close))
    w1_close = float(latest.get('w1_close', close))

    vol_window = df['volume'].iloc[-20:]
    avg_volume = float(vol_window.mean())

    bars_back_3 = df.iloc[-5]['close'] if len(df) >= 5 else df.iloc[0]['close']
    bars_back_24 = df.iloc[-26]['close'] if len(df) >= 26 else df.iloc[0]['close']

    try:
        dt = pd.Timestamp(latest['Datetime'])
        hour = dt.hour
        dow = dt.dayofweek
    except Exception:
        hour = 12
        dow = 1

    entry_price = state.get('entry_price', 0)
    position = state.get('position', 0)
    entry_time = state.get('entry_time')
    bars_held = 0
    if entry_time and position != 0:
        from datetime import datetime, timezone
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            bars_held = int((datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600)
        except Exception:
            bars_held = 0

    upnl = 0.0
    if position != 0 and entry_price > 0:
        upnl = (close - entry_price) / entry_price * position

    # Wick analysis (latest completed bar)
    bar_open = float(latest.get('open', close))
    bar_high = float(latest.get('high', close))
    bar_low = float(latest.get('low', close))
    bar_range = bar_high - bar_low + 1e-9
    upper_wick = (bar_high - max(bar_open, close)) / bar_range
    lower_wick = (min(bar_open, close) - bar_low) / bar_range
    body_pct = abs(close - bar_open) / bar_range

    # Candle patterns (latest vs previous completed bar)
    prev_open = float(prev.get('open', float(prev['close'])))
    prev_close = float(prev['close'])
    is_bearish_engulfing = (close < bar_open and
                            bar_open > prev_close and close < prev_open)
    is_bullish_engulfing = (close > bar_open and
                            bar_open < prev_close and close > prev_open)
    is_doji = body_pct < 0.1

    # FVG detection (3-bar pattern: bar 3 ago vs latest)
    bullish_fvg = False
    bearish_fvg = False
    if len(df) >= 5:
        bar_3ago = df.iloc[-4]
        bullish_fvg = float(bar_3ago.get('high', bar_3ago['close'])) < bar_low
        bearish_fvg = float(bar_3ago.get('low', bar_3ago['close'])) > bar_high

    perception = {
        "price": close,
        "price_vs_4h": (close - h4_close) / max(h4_close, 0.01),
        "price_vs_1d": (close - d1_close) / max(d1_close, 0.01),
        "price_vs_1w": (close - w1_close) / max(w1_close, 0.01),
        "candle_body": (close - float(latest.get('open', close))) / max(close, 0.01),
        "return_3bars": (close - float(bars_back_3)) / max(float(bars_back_3), 0.01),
        "return_24bars": (close - float(bars_back_24)) / max(float(bars_back_24), 0.01),
        "volume_ratio": float(latest['volume']) / max(avg_volume, 1),
        "atr_pct": calculate_atr(df) / max(close, 0.01),
        "hour": hour,
        "dow": dow,
        "position": position,
        "upnl": upnl,
        "bars_held": bars_held,
        # ICT features
        "upper_wick_pct": upper_wick,
        "lower_wick_pct": lower_wick,
        "body_pct": body_pct,
        "is_bearish_engulfing": is_bearish_engulfing,
        "is_bullish_engulfing": is_bullish_engulfing,
        "is_doji": is_doji,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
    }

    return perception


# ── LAYER 2: INTERPRETATION ───────────────────────────────────────────────────

def interpret(perception):
    conditions = {}
    narrative = []

    # Trend
    above_4h = perception['price_vs_4h'] > 0.001
    above_1d = perception['price_vs_1d'] > 0.001
    above_1w = perception['price_vs_1w'] > 0.001
    trend_score = sum([above_4h, above_1d, above_1w])

    if trend_score == 3:
        conditions['trend'] = 'STRONG_BULL'
        narrative.append(f"Price above all EMAs (4H/1D/1W) — strong uptrend")
    elif trend_score == 2:
        conditions['trend'] = 'MILD_BULL'
        narrative.append(f"Price above 2/3 EMAs — mild bullish bias")
    elif trend_score == 1:
        conditions['trend'] = 'MILD_BEAR'
        narrative.append(f"Price below 2/3 EMAs — mild bearish bias")
    else:
        conditions['trend'] = 'STRONG_BEAR'
        narrative.append(f"Price below all EMAs — strong downtrend")

    # Momentum
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

    # Volume
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

    # Volatility
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

    # Session — labels preserved as memory bucket keys, no quality differentiation
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
        narrative.append("Asia session")

    # Wick rejection
    if perception['upper_wick_pct'] > 0.6:
        conditions['wick'] = 'UPPER_REJECTION'
        narrative.append("Strong upper wick — bearish rejection")
    elif perception['lower_wick_pct'] > 0.6:
        conditions['wick'] = 'LOWER_REJECTION'
        narrative.append("Strong lower wick — bullish rejection")
    else:
        conditions['wick'] = 'NEUTRAL'

    # Candle pattern
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

    # FVG
    if perception['bullish_fvg']:
        conditions['fvg'] = 'BULLISH_FVG'
        narrative.append("Bullish FVG present — price may fill gap")
    elif perception['bearish_fvg']:
        conditions['fvg'] = 'BEARISH_FVG'
        narrative.append("Bearish FVG present — price may fill gap")
    else:
        conditions['fvg'] = 'NONE'

    return conditions, narrative


# ── LAYER 3: REASONING ENGINE ─────────────────────────────────────────────────

def reason(ppo_action, conditions, perception, memory):
    evidence_for = []
    evidence_against = []
    confidence = 0.50

    action_is_long = ppo_action == 1
    action_is_short = ppo_action == 2

    if action_is_long or action_is_short:

        # Trend alignment
        if action_is_long:
            if conditions['trend'] == 'STRONG_BULL':
                evidence_for.append("Strong uptrend — long aligned with trend")
                confidence += 0.15
            elif conditions['trend'] == 'MILD_BULL':
                evidence_for.append("Mild uptrend — long mildly aligned")
                confidence += 0.07
            elif conditions['trend'] == 'MILD_BEAR':
                evidence_against.append("Mild downtrend — long against trend")
                confidence -= 0.10
            elif conditions['trend'] == 'STRONG_BEAR':
                evidence_against.append("Strong downtrend — long strongly against trend")
                confidence -= 0.22

        if action_is_short:
            if conditions['trend'] == 'STRONG_BEAR':
                evidence_for.append("Strong downtrend — short aligned with trend")
                confidence += 0.15
            elif conditions['trend'] == 'MILD_BEAR':
                evidence_for.append("Mild downtrend — short mildly aligned")
                confidence += 0.07
            elif conditions['trend'] == 'MILD_BULL':
                evidence_against.append("Mild uptrend — short against trend")
                confidence -= 0.10
            elif conditions['trend'] == 'STRONG_BULL':
                evidence_against.append("Strong uptrend — short strongly against trend")
                confidence -= 0.22

        # Momentum
        if action_is_long and conditions['momentum'] in ['STRONG_UP', 'WEAK_UP']:
            evidence_for.append("Momentum supports long")
            confidence += 0.08
        elif action_is_long and conditions['momentum'] == 'STRONG_DOWN':
            evidence_against.append("Momentum strongly against long")
            confidence -= 0.12

        if action_is_short and conditions['momentum'] in ['STRONG_DOWN', 'WEAK_DOWN']:
            evidence_for.append("Momentum supports short")
            confidence += 0.08
        elif action_is_short and conditions['momentum'] == 'STRONG_UP':
            evidence_against.append("Momentum strongly against short")
            confidence -= 0.12

        # Volume
        if conditions['volume'] in ['HIGH', 'VERY_HIGH']:
            evidence_for.append("High volume confirms conviction")
            confidence += 0.08
        elif conditions['volume'] == 'LOW':
            evidence_against.append("Low volume — weak conviction")
            confidence -= 0.10

        # Session — no boost or penalty; all sessions evaluated equally.
        # Session label is still part of conditions for memory bucket keys.

        # Volatility
        if conditions['volatility'] == 'HIGH':
            evidence_against.append("High volatility — stop risk elevated")
            confidence -= 0.08

        # ICT: Wick confluence
        if action_is_long:
            if conditions.get('wick') == 'LOWER_REJECTION':
                evidence_for.append("Lower wick rejection confirms bullish entry")
                confidence += 0.08
            elif conditions.get('wick') == 'UPPER_REJECTION':
                evidence_against.append("Upper wick rejection contradicts long entry")
                confidence -= 0.08
        elif action_is_short:
            if conditions.get('wick') == 'UPPER_REJECTION':
                evidence_for.append("Upper wick rejection confirms short entry")
                confidence += 0.08
            elif conditions.get('wick') == 'LOWER_REJECTION':
                evidence_against.append("Lower wick rejection contradicts short entry")
                confidence -= 0.08

        # ICT: Candle pattern confluence
        candle = conditions.get('candle', 'NORMAL')
        if action_is_long and candle == 'BULLISH_ENGULF':
            evidence_for.append("Bullish engulfing confirms long")
            confidence += 0.10
        elif action_is_long and candle == 'BEARISH_ENGULF':
            evidence_against.append("Bearish engulfing contradicts long")
            confidence -= 0.10
        elif action_is_short and candle == 'BEARISH_ENGULF':
            evidence_for.append("Bearish engulfing confirms short")
            confidence += 0.10
        elif action_is_short and candle == 'BULLISH_ENGULF':
            evidence_against.append("Bullish engulfing contradicts short")
            confidence -= 0.10

        if candle == 'DOJI':
            evidence_against.append("Doji — avoid entry during indecision")
            confidence -= 0.12

        # ICT: FVG confluence
        fvg = conditions.get('fvg', 'NONE')
        if action_is_long and fvg == 'BULLISH_FVG':
            evidence_for.append("Bullish FVG supports long bias")
            confidence += 0.06
        elif action_is_short and fvg == 'BEARISH_FVG':
            evidence_for.append("Bearish FVG supports short bias")
            confidence += 0.06

        # Memory check
        veto, wr = memory.should_veto(conditions)
        if veto:
            evidence_against.append(
                f"MEMORY VETO: this setup has only {wr:.0%} WR historically"
            )
            confidence -= 0.30

        else:
            adj = memory.confidence_adjustment(conditions)
            if adj > 0.02:
                wr_val = memory.get_win_rate(conditions)
                evidence_for.append(
                    f"Memory: similar setups won {wr_val:.0%} historically"
                )
                confidence += adj
            elif adj < -0.02:
                wr_val = memory.get_win_rate(conditions)
                evidence_against.append(
                    f"Memory: similar setups won only {wr_val:.0%} historically"
                )
                confidence += adj

        # Regret adjustment — lower bar for setups agent kept wrongly rejecting
        regret_adj = memory.get_regret_adjustment(conditions)
        if regret_adj > 0:
            stats = memory.memory['condition_stats'].get(memory._make_key(conditions), {})
            evidence_for.append(
                f"Regret learning: agent missed profitable trades here "
                f"{stats.get('missed_profitable', 0)} times — lowering bar"
            )
            confidence += regret_adj

    elif ppo_action == 3:  # CLOSE
        confidence = 0.80
        evidence_for.append("PPO model signaling close")

        if perception['upnl'] < -0.015:
            evidence_for.append(f"Unrealized loss {perception['upnl']:.1%} — good to close")
            confidence += 0.10
        elif perception['upnl'] > 0.015:
            evidence_for.append(f"Locking in profit {perception['upnl']:.1%}")
            confidence += 0.05

    # Clamp confidence
    confidence = max(0.05, min(0.95, confidence))

    # Verdict
    if ppo_action in (1, 2):
        if confidence >= 0.72:
            verdict = "EXECUTE"
        elif confidence >= 0.62:
            verdict = "WEAK_EXECUTE"
        else:
            verdict = "REJECT"
    else:
        verdict = "EXECUTE" if confidence >= 0.50 else "REJECT"

    return verdict, confidence, evidence_for, evidence_against


# ── REGRET SIMULATION ────────────────────────────────────────────────────────

def simulate_missed_trade(df, rejection_bar_idx, rejected_action, sl_pct, tp_pct, max_bars=48):
    """
    Simulate what would have happened if we had taken the rejected trade.
    Returns (was_profitable, pnl_pct, exit_reason)
    """
    if rejection_bar_idx + 1 >= len(df):
        return None, 0.0, "NO_DATA"

    entry_price = float(df.iloc[rejection_bar_idx]['close'])
    direction = 1 if rejected_action == 1 else -1

    if direction == 1:
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    for i in range(1, min(max_bars + 1, len(df) - rejection_bar_idx)):
        bar = df.iloc[rejection_bar_idx + i]
        high = float(bar.get('high', bar['close']))
        low = float(bar.get('low', bar['close']))

        if direction == 1:
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                return False, pnl, "SL"
            if high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                return True, pnl, "TP"
        else:
            if high >= sl_price:
                pnl = (entry_price - sl_price) / entry_price
                return False, pnl, "SL"
            if low <= tp_price:
                pnl = (entry_price - tp_price) / entry_price
                return True, pnl, "TP"

    # Time exit
    exit_price = float(df.iloc[min(rejection_bar_idx + max_bars, len(df)-1)]['close'])
    pnl = (exit_price - entry_price) / entry_price * direction
    return pnl > 0, pnl, "TIME"


# ── LAYER 3.5: DYNAMIC RISK SIZING ───────────────────────────────────────────

def get_dynamic_sl_tp(conditions, memory):
    wr = memory.get_win_rate(conditions)
    if wr is None:
        return 0.015, 0.04
    if wr >= 0.70:
        return 0.015, 0.045
    elif wr >= 0.60:
        return 0.015, 0.04
    elif wr >= 0.55:
        return 0.025, 0.035
    else:
        return 0.015, 0.04


# ── LAYER 4: DECISION + EXPLANATION ──────────────────────────────────────────

def decide(ppo_action, conditions, perception, memory, narrative):
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}

    verdict, confidence, evidence_for, evidence_against = reason(
        ppo_action, conditions, perception, memory
    )

    lines = []
    lines.append(f"PPO Signal: {action_names.get(ppo_action, '?')}")
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

    lines.append(f"Confidence: {confidence:.0%}")
    lines.append(f"Verdict: {verdict}")

    reasoning_text = "\n".join(lines)

    return verdict, confidence, reasoning_text
