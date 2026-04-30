"""
Trade Memory System
===================
Stores and retrieves trade outcomes by market condition.
The agent reads this before every decision to learn from history.
"""

import json
import os
import csv
from datetime import datetime, timezone


class TradeMemory:
    def __init__(self, memory_file="trade_memory.json", log_file="trade_log.csv"):
        self.memory_file = memory_file
        self.log_file = log_file
        self.memory = self._load()
        self._init_log()

    def _load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file) as f:
                return json.load(f)
        return {
            "condition_stats": {},
            "total_trades": 0,
            "total_wins": 0,
            "meta": {
                "created": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
                "training_bars": 0
            }
        }

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "type", "action", "entry_price",
                    "exit_price", "pnl_pct", "pnl_usdt", "won",
                    "trend", "momentum", "volume", "volatility", "session",
                    "confidence", "verdict", "reasoning"
                ])

    def save(self):
        self.memory["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def _make_key(self, conditions):
        return f"{conditions['trend']}|{conditions['session']}|{conditions['volume']}|{conditions['momentum']}"

    def record_trade(self, conditions, action, entry_price, exit_price,
                     pnl_pct, pnl_usdt, confidence, verdict, reasoning,
                     trade_type="live", sl_pct=0.025):  # EXPECTANCY: added sl_pct param
        won = pnl_pct > 0
        key = self._make_key(conditions)

        if key not in self.memory["condition_stats"]:
            self.memory["condition_stats"][key] = {
                "wins": 0, "losses": 0, "total": 0,
                "total_pnl_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "win_pnls": [],
                "loss_pnls": [],
                "total_r": 0.0,  # EXPECTANCY
                "avg_r": 0.0,  # EXPECTANCY
                "win_r_sum": 0.0,  # EXPECTANCY
                "loss_r_sum": 0.0,  # EXPECTANCY
                "avg_win_r": 0.0,  # EXPECTANCY
                "avg_loss_r": 0.0,  # EXPECTANCY
            }

        s = self.memory["condition_stats"][key]
        s["total"] += 1
        s["total_pnl_pct"] += pnl_pct

        # EXPECTANCY: calculate and store R-multiple
        r_multiple = pnl_pct / max(sl_pct, 0.001)  # EXPECTANCY
        s.setdefault("total_r", 0.0)  # EXPECTANCY: backward compat
        s.setdefault("win_r_sum", 0.0)  # EXPECTANCY
        s.setdefault("loss_r_sum", 0.0)  # EXPECTANCY
        s["total_r"] += r_multiple  # EXPECTANCY
        s["avg_r"] = s["total_r"] / s["total"]  # EXPECTANCY

        if won:
            s["wins"] += 1
            s["win_pnls"].append(pnl_pct)
            s["avg_win_pct"] = sum(s["win_pnls"]) / len(s["win_pnls"])
            s["win_r_sum"] += r_multiple  # EXPECTANCY
            s["avg_win_r"] = s["win_r_sum"] / s["wins"]  # EXPECTANCY
        else:
            s["losses"] += 1
            s["loss_pnls"].append(pnl_pct)
            s["avg_loss_pct"] = sum(s["loss_pnls"]) / len(s["loss_pnls"])
            s["loss_r_sum"] += r_multiple  # EXPECTANCY
            s["avg_loss_r"] = s["loss_r_sum"] / s["losses"]  # EXPECTANCY

        self.memory["total_trades"] += 1
        if won:
            self.memory["total_wins"] += 1

        self.save()

        # Append to CSV log
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                trade_type,
                action,
                entry_price,
                exit_price,
                f"{pnl_pct:.4f}",
                f"{pnl_usdt:.2f}",
                won,
                conditions.get("trend", ""),
                conditions.get("momentum", ""),
                conditions.get("volume", ""),
                conditions.get("volatility", ""),
                conditions.get("session", ""),
                f"{confidence:.2f}",
                verdict,
                reasoning[:200]
            ])

    def record_missed_trade(self, conditions, was_profitable):
        key = self._make_key(conditions)
        if key not in self.memory["condition_stats"]:
            self.memory["condition_stats"][key] = {
                "wins": 0, "losses": 0, "total": 0,
                "total_pnl_pct": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "win_pnls": [], "loss_pnls": [],
                "missed_profitable": 0,
                "missed_losing": 0,
                "missed_total": 0,
                "regret_score": 0.0
            }
        s = self.memory["condition_stats"][key]
        s.setdefault("missed_profitable", 0)
        s.setdefault("missed_losing", 0)
        s.setdefault("missed_total", 0)
        s["missed_total"] += 1
        if was_profitable:
            s["missed_profitable"] += 1
        else:
            s["missed_losing"] += 1
        # Regret score: ratio of profitable misses to total misses
        # High regret = agent kept rejecting winning trades
        if s["missed_total"] >= 3:
            s["regret_score"] = s["missed_profitable"] / s["missed_total"]
        self.save()

    def get_regret_adjustment(self, conditions):
        key = self._make_key(conditions)
        stats = self.memory["condition_stats"].get(key, {})
        missed_total = stats.get("missed_total", 0)
        regret_score = stats.get("regret_score", 0.0)
        if missed_total < 3:
            return 0.0
        # If agent kept missing profitable trades, lower the bar
        if regret_score >= 0.70:
            return 0.15
        elif regret_score >= 0.60:
            return 0.10
        elif regret_score >= 0.50:
            return 0.05
        return 0.0

    def get_regret_summary(self, min_missed=3):
        rows = []
        for key, stats in self.memory["condition_stats"].items():
            missed = stats.get("missed_total", 0)
            if missed >= min_missed:
                rows.append({
                    "condition": key,
                    "regret_score": stats.get("regret_score", 0.0),
                    "missed_profitable": stats.get("missed_profitable", 0),
                    "missed_total": missed
                })
        return sorted(rows, key=lambda x: x["regret_score"], reverse=True)

    def find_similar(self, conditions):
        key = self._make_key(conditions)
        return self.memory["condition_stats"].get(key)

    def get_win_rate(self, conditions):
        stats = self.find_similar(conditions)
        if not stats or stats["total"] < 5:
            return None
        return stats["wins"] / stats["total"]

    def get_expectancy(self, conditions):  # EXPECTANCY
        """Returns expected R per trade for this condition set, or None if insufficient data."""  # EXPECTANCY
        stats = self.find_similar(conditions)  # EXPECTANCY
        if not stats or stats.get('total', 0) < 10:  # EXPECTANCY
            return None  # EXPECTANCY
        wr = stats['wins'] / stats['total']  # EXPECTANCY
        avg_win_r = stats.get('avg_win_r', 2.0)  # EXPECTANCY: default 2R
        avg_loss_r = stats.get('avg_loss_r', -1.0)  # EXPECTANCY: default -1R
        expectancy = (wr * avg_win_r) + ((1 - wr) * avg_loss_r)  # EXPECTANCY
        return expectancy  # EXPECTANCY

    def should_veto(self, conditions, min_trades=20, veto_threshold=0.40):  # raised to 0.40 (was 0.25, originally 0.35)
        stats = self.find_similar(conditions)
        if not stats or stats["total"] < min_trades:
            return False, None
        wr = stats["wins"] / stats["total"]
        if wr < veto_threshold:
            return True, wr
        return False, wr

    def confidence_adjustment(self, conditions):
        stats = self.find_similar(conditions)
        if not stats or stats["total"] < 5:
            return 0.0
        wr = stats["wins"] / stats["total"]
        # Shift confidence: 50% WR = 0 adjustment, 70% WR = +0.10, 30% WR = -0.10
        return (wr - 0.50) * 0.5

    def summarize(self, min_trades=5):
        rows = []
        for key, stats in self.memory["condition_stats"].items():
            if stats["total"] >= min_trades:
                wr = stats["wins"] / stats["total"]
                rows.append({
                    "condition": key,
                    "win_rate": wr,
                    "total": stats["total"],
                    "avg_pnl": stats["total_pnl_pct"] / stats["total"]
                })
        return sorted(rows, key=lambda x: x["win_rate"], reverse=True)

    def print_summary(self):
        rows = self.summarize()
        print(f"\n{'='*70}")
        print(f"TRADE MEMORY SUMMARY — {self.memory['total_trades']} total trades")
        print(f"Overall WR: {self.memory['total_wins']/max(1,self.memory['total_trades']):.1%}")
        print(f"{'='*70}")
        print(f"{'Condition':<50} {'WR':>6} {'Trades':>7} {'Avg PnL':>8}")
        print(f"{'-'*70}")
        for r in rows[:20]:
            print(f"{r['condition']:<50} {r['win_rate']:>6.1%} {r['total']:>7} {r['avg_pnl']:>8.3%}")
        print(f"{'='*70}\n")
