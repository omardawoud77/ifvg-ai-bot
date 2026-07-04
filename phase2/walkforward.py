"""
Phase 2 — Walk-Forward Validation Harness
=========================================
The honesty layer. The model is trained on a window, evaluated on the
NEXT out-of-sample window it has never seen, then the window rolls
forward and we repeat. Final judgement aggregates ONLY the out-of-sample
folds.

HARD DEPLOYMENT CRITERION (per symbol):
    - pooled OOS expectancy per trade > 0 (after fees)
    - at least MIN_OOS_TRADES trades pooled across folds
    - no single fold with catastrophic drawdown (> MAX_FOLD_DD)
If any fails → the model does not ship. No exceptions, no re-rolls until
it passes ("re-run until green" is just overfitting with extra steps).

This harness is model-agnostic: you hand it two callables.

    def train_fn(train_df, feats_df) -> model          # e.g. PPO .learn()
    def eval_fn(model, test_df, feats_df) -> pd.DataFrame
        # returns one row per SIMULATED TRADE with at least:
        #   ['entry_time', 'exit_time', 'side', 'pnl_pct']
        # pnl_pct must be NET of fees, matching live fee assumptions.

Usage sketch (the real train/eval plug in after the shadow-data spec):

    from walkforward import walk_forward, DEFAULT_FOLDS
    report = walk_forward(df_1h, feats, train_fn, eval_fn)
    print(report.summary())
    report.to_csv("wf_BTCUSDT.csv")
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------- criteria
MIN_OOS_TRADES = 30       # below this, expectancy is noise
MAX_FOLD_DD = -0.15       # -15% equity drawdown in any single OOS fold = fail
FEE_NOTE = ("eval_fn must apply the SAME fees/slippage as live "
            "(taker fee both sides + realistic slippage), or the whole "
            "exercise is fiction.")

# ------------------------------------------------------------- folds
@dataclass
class FoldSpec:
    train_months: int = 8
    test_months: int = 2
    step_months: int = 2   # roll forward by the test size → contiguous OOS


def make_folds(index: pd.DatetimeIndex, spec: FoldSpec):
    """Yield (train_mask, test_mask) boolean arrays over the bar index."""
    start, end = index.min(), index.max()
    t0 = start
    while True:
        train_end = t0 + pd.DateOffset(months=spec.train_months)
        test_end = train_end + pd.DateOffset(months=spec.test_months)
        if test_end > end:
            break
        train_mask = (index >= t0) & (index < train_end)
        test_mask = (index >= train_end) & (index < test_end)
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            yield train_mask, test_mask, (train_end, test_end)
        t0 = t0 + pd.DateOffset(months=spec.step_months)


# ------------------------------------------------------------- metrics
def fold_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "expectancy": np.nan, "win_rate": np.nan,
                "profit_factor": np.nan, "max_dd": 0.0,
                "long_share": np.nan}
    pnl = trades["pnl_pct"].astype(float)
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {
        "trades": len(trades),
        "expectancy": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "max_dd": max_dd,
        "long_share": float((trades["side"].str.upper() == "LONG").mean())
        if "side" in trades else np.nan,
    }


# ------------------------------------------------------------- report
@dataclass
class WFReport:
    symbol: str
    folds: list = field(default_factory=list)     # list of dicts
    all_trades: list = field(default_factory=list)

    @property
    def pooled(self) -> dict:
        trades = (pd.concat(self.all_trades, ignore_index=True)
                  if self.all_trades else pd.DataFrame())
        return fold_metrics(trades)

    def passes_deployment(self) -> tuple[bool, list[str]]:
        p = self.pooled
        reasons = []
        if p["trades"] < MIN_OOS_TRADES:
            reasons.append(f"only {p['trades']} OOS trades (<{MIN_OOS_TRADES})")
        if not (p["expectancy"] > 0):
            reasons.append(f"pooled OOS expectancy {p['expectancy']:.4%} ≤ 0")
        worst = min((f["max_dd"] for f in self.folds), default=0.0)
        if worst < MAX_FOLD_DD:
            reasons.append(f"worst fold drawdown {worst:.1%} < {MAX_FOLD_DD:.0%}")
        return (len(reasons) == 0), reasons

    def summary(self) -> str:
        lines = [f"Walk-forward report — {self.symbol}",
                 f"folds: {len(self.folds)}"]
        for i, f in enumerate(self.folds, 1):
            lines.append(
                f"  fold {i} [{f['test_start']:%Y-%m} → {f['test_end']:%Y-%m}]"
                f"  trades={f['trades']:>3}  exp={f['expectancy']:+.3%}"
                f"  wr={f['win_rate']:.0%}  pf={f['profit_factor']:.2f}"
                f"  dd={f['max_dd']:.1%}  long%={f['long_share']:.0%}"
                if f["trades"] else f"  fold {i}: no trades")
        p = self.pooled
        ok, reasons = self.passes_deployment()
        lines.append(f"POOLED OOS: trades={p['trades']} "
                     f"exp={p['expectancy']:+.3%} wr={p['win_rate']:.0%} "
                     f"pf={p['profit_factor']:.2f}")
        lines.append("DEPLOYMENT: " + ("PASS ✅" if ok else
                                       "FAIL ❌ — " + "; ".join(reasons)))
        return "\n".join(lines)

    def to_csv(self, path: str):
        pd.DataFrame(self.folds).to_csv(path, index=False)


# ------------------------------------------------------------- driver
def walk_forward(df: pd.DataFrame, feats: pd.DataFrame,
                 train_fn, eval_fn, symbol: str = "?",
                 spec: FoldSpec = FoldSpec()) -> WFReport:
    """df must carry an 'open_time' column (UTC). feats row-aligned to df."""
    index = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True))
    report = WFReport(symbol=symbol)
    for k, (tr, te, (test_start, test_end)) in enumerate(
            make_folds(index, spec), 1):
        print(f"[{symbol}] fold {k}: train {tr.sum()} bars, "
              f"test {te.sum()} bars ({test_start:%Y-%m} → {test_end:%Y-%m})")
        model = train_fn(df[tr], feats[tr])
        trades = eval_fn(model, df[te], feats[te])
        m = fold_metrics(trades)
        m.update(test_start=test_start, test_end=test_end)
        report.folds.append(m)
        if trades is not None and not trades.empty:
            report.all_trades.append(trades)
    return report


# ------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Smoke test with synthetic data + a dummy strategy, so the harness
    # itself can be verified before any real model exists.
    rng = np.random.default_rng(7)
    n = 24 * 365 * 2
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"open_time": ts, "close": price,
                       "high": price * 1.002, "low": price * 0.998})
    feats = pd.DataFrame({"noise": rng.normal(size=n)})

    def train_fn(train_df, train_feats):
        return None  # coin-flip "model"

    def eval_fn(model, test_df, test_feats):
        m = len(test_df)
        entries = rng.choice(m - 48, size=max(m // 200, 5), replace=False)
        rows = []
        for e in sorted(entries):
            side = rng.choice(["LONG", "SHORT"])
            r = (test_df["close"].iloc[e + 48] / test_df["close"].iloc[e]) - 1
            pnl = (r if side == "LONG" else -r) - 0.0008  # fees
            rows.append({"entry_time": test_df["open_time"].iloc[e],
                         "side": side, "pnl_pct": pnl})
        return pd.DataFrame(rows)

    rep = walk_forward(df, feats, train_fn, eval_fn, symbol="SYNTH")
    print()
    print(rep.summary())
    print("\n(A coin-flip strategy should FAIL deployment — that's the point.)")
