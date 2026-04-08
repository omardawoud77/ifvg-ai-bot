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
                     pnl_pct, pnl_usdt, confidence, verdict, reasoning, trade_type="live"):
        won = pnl_pct > 0
        key = self._make_key(conditions)

        if key not in self.memory["condition_stats"]:
            self.memory["condition_stats"][key] = {
                "wins": 0, "losses": 0, "total": 0,
                "total_pnl_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "win_pnls": [],
                "loss_pnls": []
            }

        s = self.memory["condition_stats"][key]
        s["total"] += 1
        s["total_pnl_pct"] += pnl_pct

        if won:
            s["wins"] += 1
            s["win_pnls"].append(pnl_pct)
            s["avg_win_pct"] = sum(s["win_pnls"]) / len(s["win_pnls"])
        else:
            s["losses"] += 1
            s["loss_pnls"].append(pnl_pct)
            s["avg_loss_pct"] = sum(s["loss_pnls"]) / len(s["loss_pnls"])

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

    def find_similar(self, conditions):
        key = self._make_key(conditions)
        return self.memory["condition_stats"].get(key)

    def get_win_rate(self, conditions):
        stats = self.find_similar(conditions)
        if not stats or stats["total"] < 5:
            return None
        return stats["wins"] / stats["total"]

    def should_veto(self, conditions, min_trades=10, veto_threshold=0.45):
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
