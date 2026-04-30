"""
Position Auditor Agent
======================
Runs every cycle when the bot is in an open trade.
Re-evaluates the trade thesis and issues EXIT_NOW if conditions have
degraded significantly since entry. This catches situations like:
  - Entered LONG, regime flipped to TRENDING_BEAR
  - Entered SHORT, sentiment turned BULLISH strongly
  - Trade held too long with no progress toward TP
  - Consecutive adverse bars eroding the original setup

The auditor does NOT replace SL/TP — it's an intelligence layer on top.
It asks: "If I had to enter this trade right now, would I?"
If the answer is strongly NO, it recommends EXIT.

Output:
  {
    'action':    'HOLD' | 'EXIT_NOW' | 'REDUCE',
    'reason':    str,
    'urgency':   'LOW' | 'MEDIUM' | 'HIGH',
    'score':     float   # -1 (exit now) to +1 (hold strong)
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
EXIT_SCORE_THRESHOLD   = -0.35   # below this → EXIT_NOW
REDUCE_SCORE_THRESHOLD = -0.15   # below this → REDUCE (not implemented in agent yet, reserved)
MAX_BARS_NO_PROGRESS   = 24      # hours — if no TP progress in 24h, flag it
ADVERSE_BAR_THRESHOLD  = 3       # consecutive bars moving against position → flag


class PositionAuditor:
    """
    Call audit(state, conditions, perception, sentiment_signal) every cycle
    when position != 0. Returns action dict.
    """

    def audit(self, state: dict, conditions: dict, perception: dict,
              sentiment_signal: dict = None) -> dict:
        """
        state:            current symbol state dict
        conditions:       from reasoning_engine.interpret()
        perception:       from reasoning_engine.perceive()
        sentiment_signal: from SentimentAgent.get_signal() (optional)

        Returns action dict.
        """
        position = state.get('position', 0)
        if position == 0:
            return self._hold("No open position")

        entry_price    = state.get('entry_price', 0)
        tp_pct         = state.get('tp_pct', 0.05)
        sl_pct         = state.get('sl_pct', 0.025)
        entry_time_str = state.get('entry_time')
        entry_conds    = state.get('entry_conditions', {})
        upnl           = perception.get('upnl', 0.0)
        current_price  = perception.get('price', 0)

        reasons     = []
        score       = 0.0   # starts neutral, goes negative = exit signal

        is_long  = position == 1
        is_short = position == -1

        # ── CHECK 1: Regime flip since entry ─────────────────────────────────
        entry_regime   = entry_conds.get('regime', 'UNKNOWN')
        current_regime = conditions.get('regime', 'UNKNOWN')

        if is_long and current_regime == 'TRENDING_BEAR':
            reasons.append(f"REGIME FLIP: entered in {entry_regime}, now TRENDING_BEAR — thesis broken")
            score -= 0.50   # strong exit signal

        elif is_short and current_regime == 'TRENDING_BULL':
            reasons.append(f"REGIME FLIP: entered in {entry_regime}, now TRENDING_BULL — thesis broken")
            score -= 0.50

        elif entry_regime != current_regime and current_regime in ('HIGH_VOLATILITY', 'LOW_QUALITY'):
            reasons.append(f"Regime degraded from {entry_regime} → {current_regime}")
            score -= 0.20

        # ── CHECK 2: Trend direction flip ─────────────────────────────────────
        entry_trend   = entry_conds.get('trend', 'UNKNOWN')
        current_trend = conditions.get('trend', 'UNKNOWN')

        if is_long and current_trend in ('STRONG_BEAR',) and entry_trend != 'STRONG_BEAR':
            reasons.append(f"Trend flipped to STRONG_BEAR while holding LONG")
            score -= 0.30

        elif is_short and current_trend in ('STRONG_BULL',) and entry_trend != 'STRONG_BULL':
            reasons.append(f"Trend flipped to STRONG_BULL while holding SHORT")
            score -= 0.30

        # ── CHECK 3: Momentum against position ───────────────────────────────
        current_mom = conditions.get('momentum', 'NEUTRAL')

        if is_long and current_mom == 'STRONG_DOWN':
            reasons.append("Strong downward momentum developing against LONG")
            score -= 0.15

        elif is_short and current_mom == 'STRONG_UP':
            reasons.append("Strong upward momentum developing against SHORT")
            score -= 0.15

        # ── CHECK 4: Time-based — no progress toward TP ───────────────────────
        bars_held = self._bars_held(entry_time_str)
        if bars_held > MAX_BARS_NO_PROGRESS:
            tp_progress = upnl / tp_pct if tp_pct > 0 else 0
            if tp_progress < 0.25:   # less than 25% of TP achieved in 24h
                reasons.append(
                    f"Held {bars_held}h with only {tp_progress:.0%} TP progress — stale trade"
                )
                score -= 0.25
            elif tp_progress < 0:
                reasons.append(f"Held {bars_held}h and in loss — time to review")
                score -= 0.20

        # ── CHECK 5: Sentiment reversal ───────────────────────────────────────
        if sentiment_signal:
            s_dir      = sentiment_signal.get('direction', 'NEUTRAL')
            s_strength = float(sentiment_signal.get('strength', 0))
            fg_value   = int(sentiment_signal.get('fear_greed_value', 50))

            if is_long and s_dir == 'BEARISH' and s_strength > 0.65:
                reasons.append(
                    f"Strong BEARISH sentiment ({s_strength:.0%}) against LONG position"
                )
                score -= 0.20

            elif is_short and s_dir == 'BULLISH' and s_strength > 0.65:
                reasons.append(
                    f"Strong BULLISH sentiment ({s_strength:.0%}) against SHORT position"
                )
                score -= 0.20

            # Extreme fear/greed — contrarian signal
            if fg_value <= 15 and is_short:
                reasons.append(f"Extreme Fear (F&G={fg_value}) — shorts may be crowded, watch for squeeze")
                score -= 0.10
            elif fg_value >= 85 and is_long:
                reasons.append(f"Extreme Greed (F&G={fg_value}) — longs may be crowded, watch for flush")
                score -= 0.10

        # ── CHECK 6: Funding rate squeeze risk ───────────────────────────────
        if sentiment_signal:
            funding = float(sentiment_signal.get('funding_rate', 0))
            if is_long and funding > 0.0015:
                reasons.append(
                    f"Very high funding rate {funding:+.4f} — long positions crowded, squeeze risk"
                )
                score -= 0.12
            elif is_short and funding < -0.0015:
                reasons.append(
                    f"Very negative funding {funding:+.4f} — short positions crowded, squeeze risk"
                )
                score -= 0.12

        # ── CHECK 7: Positive holders — reinforce if thesis still intact ──────
        if is_long and current_regime in ('TRENDING_BULL', 'RANGING') and upnl > 0:
            reasons.append("Regime intact and position profitable — hold")
            score += 0.15

        elif is_short and current_regime in ('TRENDING_BEAR', 'RANGING') and upnl > 0:
            reasons.append("Regime intact and position profitable — hold")
            score += 0.15

        if is_long and current_trend in ('STRONG_BULL', 'MILD_BULL') and current_mom in ('STRONG_UP', 'WEAK_UP'):
            score += 0.10

        elif is_short and current_trend in ('STRONG_BEAR', 'MILD_BEAR') and current_mom in ('STRONG_DOWN', 'WEAK_DOWN'):
            score += 0.10

        # ── Clamp and decide ──────────────────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score <= EXIT_SCORE_THRESHOLD:
            urgency = 'HIGH' if score <= -0.60 else 'MEDIUM'
            reason_str = " | ".join(reasons) if reasons else "Multiple thesis failures"
            log.warning(
                f"[AUDITOR] EXIT_NOW signal | score={score:.2f} | urgency={urgency} | "
                f"position={'LONG' if is_long else 'SHORT'} @ ${entry_price:,.2f} | "
                f"uPnL={upnl*100:+.2f}% | reasons: {reason_str}"
            )
            return {
                'action':  'EXIT_NOW',
                'reason':  reason_str,
                'urgency': urgency,
                'score':   round(score, 3),
            }

        if score <= REDUCE_SCORE_THRESHOLD:
            reason_str = " | ".join(reasons) if reasons else "Setup degrading"
            log.info(f"[AUDITOR] REDUCE signal | score={score:.2f} | reasons: {reason_str}")
            return {
                'action':  'REDUCE',
                'reason':  reason_str,
                'urgency': 'LOW',
                'score':   round(score, 3),
            }

        # All good — hold
        hold_reasons = [r for r in reasons if "hold" in r.lower() or "intact" in r.lower()]
        positive_msg = " | ".join(hold_reasons) if hold_reasons else "Thesis intact"
        if reasons and not hold_reasons:
            positive_msg = "Minor concerns but within tolerance"

        return self._hold(positive_msg, score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hold(self, reason: str = "Thesis intact", score: float = 0.0) -> dict:
        return {
            'action':  'HOLD',
            'reason':  reason,
            'urgency': 'LOW',
            'score':   round(score, 3),
        }

    def _bars_held(self, entry_time_str: str | None) -> int:
        """Returns hours held since entry_time_str (ISO format)."""
        if not entry_time_str:
            return 0
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            delta    = datetime.now(timezone.utc) - entry_dt
            return int(delta.total_seconds() / 3600)
        except Exception:
            return 0
