"""
Crypto MTF Balanced Training — Bear-balanced + Expectancy-optimized
====================================================================
Retraining script that addresses the model's bull-bias by:
  1. Halving hold_cost (less pressure to over-trade)
  2. Boosting short-win rewards by 0.5 vs long-win rewards
  3. Duplicating bear segments in training data 2x (3x total exposure)
  4. Selecting best checkpoint by EXPECTANCY, not WR
  5. True 70/15/15 train/val/test split with separate test holdout
  6. Tracking bull/bear WR separately during evaluation

Run:
    cd /Users/moura/trading-indicator/ai-model/crypto
    python3 rl_train_balanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from crypto_env_v2 import CryptoTechEnv

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PKL  = "btc_historical_mtf.pkl"
MODEL_PATH   = "crypto_mtf_balanced_best"   # .zip appended by SB3
LOG_PATH     = "rl_train_balanced.log"
REPORT_PATH  = "training_report.json"

TOTAL_STEPS  = 4_000_000
EVAL_EVERY   = 50_000
PATIENCE     = 500_000

MIN_PROFIT_PCT  = 0.025   # 2.5% threshold for "real" win
MIN_TRADES_EVAL = 50
MIN_EXPECTANCY  = 0.05    # require >0.05R per trade for valid checkpoint

EMA_PERIOD      = 50
BEAR_MIN_RUN    = 5       # min consecutive bars for a "bear segment"
BEAR_DUPLICATES = 2       # extra copies (so original + 2 dupes = 3x total)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Custom Env: Balanced Reward + Bull/Bear Tracking ──────────────────────────

class CryptoBalancedEnv(CryptoTechEnv):
    """
    CryptoTechEnv with:
      - hold_cost halved (0.0001)
      - Long win  (pnl >= 2.5%): +3.0
      - Short win (pnl >= 2.5%): +3.5  (extra bonus to fix bull bias)
      - Small win (0 to 2.5%):   -0.3
      - Loss:                    -2.0
      - Per-trade tracking of entry direction + entry trend for stratified WR

    Implementation notes:
      - The categorical close reward REPLACES the parent's `pnl_pct * 5`
        contribution but PRESERVES hold_cost, continuous unrealized-PnL
        gradient, timeout penalty, and the end-of-episode bonus
        (pnl_pct * 10 + wr * 2). This is done by subtracting the parent's
        close bonus before adding the categorical.
      - Entry trend (bull/bear) is labelled at entry time using EMA50 on
        the close series of the env's df.
    """

    def __init__(self, df, **kwargs):
        # Halve hold cost — less pressure to over-trade
        kwargs.setdefault('hold_cost', 0.0001)
        super().__init__(df, **kwargs)

        # Pre-compute EMA50 on close so we can label trades bull/bear
        self.ema50 = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean().values

        # Per-episode trade ledger for stratified eval
        self.trade_ledger = []   # list of dicts: {direction, entry_trend, pnl_pct}
        self._entry_direction = 0
        self._entry_trend = 'unknown'

    def _reset_state(self):
        super()._reset_state()
        self.trade_ledger = []
        self._entry_direction = 0
        self._entry_trend = 'unknown'

    def step(self, action):
        # Capture pre-step position to detect entries
        pre_position = self.position

        # Run parent step (handles obs, hold cost, SL, TP, continuous gradient,
        # close bonus pnl_pct*5, timeout penalty, end-of-episode bonus, etc.)
        obs, reward, done, truncated, info = super().step(action)

        # Detect entry — position went from 0 to non-zero
        if pre_position == 0 and self.position != 0:
            self._entry_direction = self.position
            # Parent increments current_idx at end of step, so the entry bar
            # is one before the new current_idx.
            entry_idx = max(0, min(self.current_idx - 1, len(self.ema50) - 1))
            close_at_entry = float(self.df['close'].iloc[entry_idx])
            ema_at_entry = float(self.ema50[entry_idx])
            self._entry_trend = 'bull' if close_at_entry >= ema_at_entry else 'bear'

        # On trade close, override the close-bonus and record ledger entry
        if info.get('trade_closed', False):
            pnl_pct = info.get('pnl_pct', 0.0)

            # Remove parent's pnl_pct*5 close-bonus contribution so we can
            # replace it with our categorical reward without losing the
            # end-of-episode bonus (which the parent also adds on the last bar).
            reward -= pnl_pct * 5

            if pnl_pct >= MIN_PROFIT_PCT:
                if self._entry_direction == -1:
                    reward += 3.5    # SHORT win bonus
                else:
                    reward += 3.0    # LONG win
            elif pnl_pct >= 0:
                reward += -0.3       # small win penalty (anti-scalp)
            else:
                reward += -2.0       # loss

            # Record for stratified analysis
            self.trade_ledger.append({
                'direction': self._entry_direction,
                'entry_trend': self._entry_trend,
                'pnl_pct': pnl_pct,
            })

            # Reset trade-state trackers
            self._entry_direction = 0
            self._entry_trend = 'unknown'

        return obs, reward, done, truncated, info


# ── Bear-balancing helpers ────────────────────────────────────────────────────

def find_bear_segments(df, ema_period=EMA_PERIOD, min_run=BEAR_MIN_RUN):
    """Identify contiguous runs of bars where close < EMA(close).
    Returns a list of (start_idx, end_idx_inclusive) tuples for runs
    of length >= min_run."""
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().values
    is_bear = df['close'].values < ema
    segments = []
    i = 0
    n = len(df)
    while i < n:
        if is_bear[i]:
            j = i
            while j < n and is_bear[j]:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                segments.append((i, j - 1))
            i = j
        else:
            i += 1
    return segments


def build_balanced_train_df(train_df, n_duplicates=BEAR_DUPLICATES):
    """Append shuffled bear segments to the training set to balance
    bull/bear exposure.

    Each bear segment is duplicated `n_duplicates` times. Within each
    segment the bars stay in temporal order (preserves MTF context).
    The order in which segments are appended is randomized so the agent
    doesn't learn a positional artifact from the duplicates.

    The original training data is left untouched at the front of the
    output dataframe — its temporal structure is preserved.
    """
    segments = find_bear_segments(train_df)
    total_bear_bars = sum(end - start + 1 for start, end in segments)

    if not segments:
        log.warning("⚠️  No bear segments found — returning train_df unchanged")
        return train_df.reset_index(drop=True)

    log.info(f"📉 Found {len(segments)} bear segments ({total_bear_bars} bars total)")

    rng = np.random.default_rng(seed=42)
    all_dupes = []
    for _ in range(n_duplicates):
        seg_order = list(range(len(segments)))
        rng.shuffle(seg_order)
        for seg_idx in seg_order:
            start, end = segments[seg_idx]
            all_dupes.append(train_df.iloc[start:end + 1].copy())

    duped = pd.concat(all_dupes, ignore_index=True)
    log.info(f"🐻 Adding {len(duped)} duplicated bear-segment bars "
             f"({n_duplicates}x copies, shuffled segment order)")

    balanced = pd.concat([train_df, duped], ignore_index=True)
    log.info(f"📊 Balanced train: {len(train_df)} original + {len(duped)} dupes "
             f"= {len(balanced)} total bars")
    return balanced


# ── Eval helpers ──────────────────────────────────────────────────────────────

def evaluate_with_stratification(model, eval_env):
    """Run a full deterministic eval episode and return aggregate stats
    including bull/bear stratified WR and expectancy in R-multiples
    (R = MIN_PROFIT_PCT)."""
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = eval_env.step(action)

    ledger = list(eval_env.trade_ledger)

    pnl     = info.get('total_pnl', 0)
    pnl_pct = info.get('total_pnl_pct', 0)
    wr      = info.get('win_rate', 0)
    wins    = info.get('wins', 0)
    losses  = info.get('losses', 0)
    dd      = info.get('max_drawdown', 0)
    trades  = info.get('total_trades', 0)

    # Expectancy in R-multiples (1R = MIN_PROFIT_PCT)
    R = MIN_PROFIT_PCT
    win_rs  = [(t['pnl_pct'] / R) for t in ledger if t['pnl_pct'] > 0]
    loss_rs = [(t['pnl_pct'] / R) for t in ledger if t['pnl_pct'] <= 0]
    avg_win_r  = float(np.mean(win_rs))  if win_rs  else 0.0
    avg_loss_r = float(np.mean(loss_rs)) if loss_rs else 0.0
    expectancy = (wr * avg_win_r) + ((1.0 - wr) * avg_loss_r)

    # Stratified bull/bear WR
    bull_trades = [t for t in ledger if t['entry_trend'] == 'bull']
    bear_trades = [t for t in ledger if t['entry_trend'] == 'bear']
    bull_wins = sum(1 for t in bull_trades if t['pnl_pct'] > 0)
    bear_wins = sum(1 for t in bear_trades if t['pnl_pct'] > 0)
    bull_wr = bull_wins / len(bull_trades) if bull_trades else 0.0
    bear_wr = bear_wins / len(bear_trades) if bear_trades else 0.0

    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'wr': wr,
        'wins': wins,
        'losses': losses,
        'dd': dd,
        'trades': trades,
        'avg_win_r': avg_win_r,
        'avg_loss_r': avg_loss_r,
        'expectancy': expectancy,
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_wr': bull_wr,
        'bear_wr': bear_wr,
    }


# ── Callback ──────────────────────────────────────────────────────────────────

class BalancedCallback(BaseCallback):
    """
    Eval callback that selects best checkpoint by EXPECTANCY (not WR or PnL).
    Tracks bull/bear WR separately every eval. Implements patience-based
    early stop.
    """

    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env
        self.best_expectancy = -1e9
        self.best_eval = None
        self.steps_since_best = 0

    def _on_step(self):
        if self.n_calls % EVAL_EVERY != 0:
            return True

        stats = evaluate_with_stratification(self.model, self.eval_env)

        log.info(
            f"  Step {self.n_calls:>9,} | "
            f"WR: {stats['wr']:.1%} ({stats['wins']}W/{stats['losses']}L) | "
            f"Expect: {stats['expectancy']:+.2f}R | "
            f"Trades: {stats['trades']} | "
            f"Bull WR: {stats['bull_wr']:.1%} ({stats['bull_trades']}) | "
            f"Bear WR: {stats['bear_wr']:.1%} ({stats['bear_trades']}) | "
            f"PnL: ${stats['pnl']:+.2f}"
        )

        # Selection: best by expectancy, with min trades + min expectancy floor
        qualifies = (
            stats['trades'] >= MIN_TRADES_EVAL
            and stats['expectancy'] > MIN_EXPECTANCY
        )
        if qualifies and stats['expectancy'] > self.best_expectancy:
            self.best_expectancy = stats['expectancy']
            self.best_eval = dict(stats)
            self.steps_since_best = 0
            self.model.save(MODEL_PATH)
            log.info(
                f"  💾 New best | Expect: {stats['expectancy']:+.2f}R | "
                f"WR: {stats['wr']:.1%} | Trades: {stats['trades']}"
            )
        else:
            self.steps_since_best += EVAL_EVERY

        if self.steps_since_best >= PATIENCE:
            log.info(
                f"\n⏹️  Early stop — no expectancy improvement for "
                f"{PATIENCE:,} steps (best: {self.best_expectancy:+.2f}R)"
            )
            return False

        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DATASET_PKL):
        log.error(f"❌ Dataset not found: {DATASET_PKL}")
        log.error("   Run historical_trainer.py first to generate the dataset.")
        sys.exit(1)

    log.info(f"📥 Loading dataset: {DATASET_PKL}")
    df = pd.read_pickle(DATASET_PKL)
    log.info(f"✅ {len(df)} bars | {df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]}")

    # Three-way time-based split: 70 / 15 / 15
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)
    log.info(
        f"📊 Train: {len(train_df):,} bars | "
        f"Val: {len(val_df):,} bars | "
        f"Test (holdout): {len(test_df):,} bars"
    )

    # Bear-balance the training set
    log.info("\n🐻 Bear-balancing training data...")
    balanced_train_df = build_balanced_train_df(train_df, n_duplicates=BEAR_DUPLICATES)

    # Build envs — train uses balanced data, val/test use raw chronological splits
    log.info("\n🏗️  Building environments...")
    train_env = CryptoBalancedEnv(balanced_train_df)
    val_env   = CryptoBalancedEnv(val_df)
    test_env  = CryptoBalancedEnv(test_df)
    log.info(f"   Reward: +3.0 long-win | +3.5 short-win | -0.3 small-win | -2.0 loss")
    log.info(f"   Hold cost: {train_env.hold_cost} (halved from baseline)")
    log.info(f"   MIN_PROFIT_PCT: {MIN_PROFIT_PCT:.1%}")

    log.info("\n🤖 Building PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    log.info(f"   Network: 44 → 256 → 256 → 128 → 4 | Params: {n_params:,}")

    log.info(f"\n🚀 Training up to {TOTAL_STEPS:,} steps...")
    log.info(f"   Eval every {EVAL_EVERY:,} (on val set) | Patience {PATIENCE:,}")
    log.info(f"   Selection: best EXPECTANCY (>={MIN_EXPECTANCY:.2f}R, "
             f">={MIN_TRADES_EVAL} trades)")
    log.info("=" * 80)

    callback = BalancedCallback(val_env)
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callback,
        progress_bar=False,
    )

    # ── Final eval on TRUE holdout test set ──
    log.info("\n" + "=" * 80)
    log.info("FINAL EVALUATION ON HOLDOUT TEST SET")
    log.info("=" * 80)

    if not os.path.exists(MODEL_PATH + ".zip"):
        log.error("⚠️  No best checkpoint saved — callback never found a qualifying eval")
        log.error("   (either MIN_TRADES floor wasn't met or expectancy never crossed "
                  f"{MIN_EXPECTANCY:.2f}R)")
        return

    best = PPO.load(MODEL_PATH)
    final_stats = evaluate_with_stratification(best, test_env)

    log.info(f"\n📊 BALANCED MODEL — HOLDOUT TEST:")
    log.info(f"   PnL:        ${final_stats['pnl']:+.2f} ({final_stats['pnl_pct']:+.1%})")
    log.info(f"   Win Rate:   {final_stats['wr']:.1%} "
             f"({final_stats['wins']}W / {final_stats['losses']}L)")
    log.info(f"   Expectancy: {final_stats['expectancy']:+.2f}R per trade")
    log.info(f"   Trades:     {final_stats['trades']}")
    log.info(f"   Max DD:     {final_stats['dd']:.1%}")
    log.info(f"   Bull WR:    {final_stats['bull_wr']:.1%} "
             f"({final_stats['bull_trades']} trades)")
    log.info(f"   Bear WR:    {final_stats['bear_wr']:.1%} "
             f"({final_stats['bear_trades']} trades)")
    log.info(f"   Avg Win R:  {final_stats['avg_win_r']:+.2f}")
    log.info(f"   Avg Loss R: {final_stats['avg_loss_r']:+.2f}")

    # Save report
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_path': MODEL_PATH + '.zip',
        'dataset': DATASET_PKL,
        'total_steps': TOTAL_STEPS,
        'hyperparameters': {
            'learning_rate': 2e-4,
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'net_arch': [256, 256, 128],
        },
        'reward_config': {
            'hold_cost': 0.0001,
            'long_win': 3.0,
            'short_win': 3.5,
            'small_win': -0.3,
            'loss': -2.0,
            'min_profit_pct': MIN_PROFIT_PCT,
        },
        'data_split': {
            'train_bars': len(train_df),
            'balanced_train_bars': len(balanced_train_df),
            'val_bars': len(val_df),
            'test_bars': len(test_df),
        },
        'best_val_eval': callback.best_eval,
        'holdout_test': final_stats,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\n💾 Saved: {MODEL_PATH}.zip")
    log.info(f"💾 Report: {REPORT_PATH}")

    if final_stats['expectancy'] >= 0.20:
        log.info(f"✅ Expectancy {final_stats['expectancy']:+.2f}R — strong, deploy candidate")
    elif final_stats['expectancy'] >= 0.05:
        log.info(f"🟡 Expectancy {final_stats['expectancy']:+.2f}R — positive but marginal")
    else:
        log.info(f"❌ Expectancy {final_stats['expectancy']:+.2f}R — not viable")


if __name__ == "__main__":
    main()
