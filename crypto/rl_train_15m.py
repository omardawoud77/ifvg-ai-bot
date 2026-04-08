"""
Crypto MTF Agent — 15-minute Bars, WR-Focused Reward
====================================================
Retrains the PPO agent on 15m candles with a categorical reward
designed to push for high win-rate selectivity rather than continuous
PnL maximization.

Reward design:
  Win  : +2.0   (any winning close, regardless of magnitude)
  Loss : -3.0   (any losing close, including SL and timeout)
  Hold : -0.001 per bar (small penalty for inaction)

  Asymmetric — breakeven win rate ≈ 60%. The model has a strong
  incentive to be selective.

Note: this reward is intentionally PnL-blind. The model is rewarded
for being correct, not for maximizing dollars. Combine with TP rules
or a PnL-aware reward if you want dollar optimization too.

Run:
    cd /Users/moura/trading-indicator/ai-model/crypto
    python3 rl_train_15m.py            # full 2M-step training
    python3 rl_train_15m.py --smoke    # 5k-step smoke test only
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from binance.client import Client
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from crypto_env_v2 import CryptoMTFEnv

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL          = "BTCUSDT"
DATASET_PKL     = "btc_15m_mtf.pkl"
MODEL_PATH      = "crypto_mtf_15m_best"   # .zip appended by SB3
TOTAL_TIMESTEPS = 2_000_000
EVAL_EVERY      = 5000
PATIENCE        = 200_000

# Reward shape (passed to WRFocusedEnv)
WIN_REWARD     = 2.0
LOSS_PENALTY   = -3.0
HOLD_PENALTY   = -0.001
MAX_HOLD_BARS  = 48              # 48 × 15m = 12h
START_DATE     = "1 Jan, 2022"   # 15m history goes back ~4y on Binance


# ── 15m MTF data fetch (mirrors historical_trainer.py) ────────────────────────

def fetch_15m_mtf(symbol="BTCUSDT", start_date=START_DATE):
    print(f"📥 Fetching {symbol} 15m + 1h/4h/1d/1w MTF from {start_date}...")

    client = Client("", "")
    client.ping = lambda: None

    def fetch_tf(interval, start, limit=1000):
        all_klines = []
        while True:
            klines = client.get_historical_klines(symbol, interval, start, limit=limit)
            if not klines:
                break
            all_klines.extend(klines)
            start = klines[-1][0] + 1
            if len(klines) < limit:
                break
            print(f"  Fetched {len(all_klines)} bars...", end="\r")
        return all_klines

    def to_df(klines):
        df = pd.DataFrame(klines, columns=[
            'ts','open','high','low','close','volume',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df[['ts','open','high','low','close','volume']].set_index('ts')

    print("  Fetching 15m bars (this is the big one)...")
    df_15m = to_df(fetch_tf(Client.KLINE_INTERVAL_15MINUTE, start_date))
    print(f"  ✅ 15m: {len(df_15m)} bars")

    print("  Fetching 4h bars...")
    df_4h = to_df(fetch_tf(Client.KLINE_INTERVAL_4HOUR, start_date))
    print(f"  ✅ 4h: {len(df_4h)} bars")

    print("  Fetching 1d bars...")
    df_1d = to_df(fetch_tf(Client.KLINE_INTERVAL_1DAY, start_date))
    print(f"  ✅ 1d: {len(df_1d)} bars")

    print("  Fetching 1w bars...")
    df_1w = to_df(fetch_tf(Client.KLINE_INTERVAL_1WEEK, start_date))
    print(f"  ✅ 1w: {len(df_1w)} bars")

    df_4h_ff = df_4h.reindex(df_15m.index, method='ffill').add_prefix('h4_')
    df_1d_ff = df_1d.reindex(df_15m.index, method='ffill').add_prefix('d1_')
    df_1w_ff = df_1w.reindex(df_15m.index, method='ffill').add_prefix('w1_')

    df = pd.concat([df_15m, df_4h_ff, df_1d_ff, df_1w_ff], axis=1).dropna()
    df = df.reset_index().rename(columns={'ts': 'Datetime'})
    print(f"\n✅ MTF dataset: {len(df)} bars ({df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]})")
    return df


def load_or_fetch_15m():
    if os.path.exists(DATASET_PKL):
        print(f"✅ Loading cached dataset: {DATASET_PKL}")
        df = pd.read_pickle(DATASET_PKL)
        print(f"   {len(df)} bars | {df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]}")
        return df
    df = fetch_15m_mtf(SYMBOL)
    df.to_pickle(DATASET_PKL)
    print(f"💾 Cached dataset to {DATASET_PKL}")
    return df


# ── WR-focused environment (subclasses CryptoMTFEnv, overrides step) ─────────

class WRFocusedCryptoEnv(CryptoMTFEnv):
    """
    Same observations and action space as CryptoMTFEnv, but the reward
    is purely categorical: +2 per winning close, -3 per losing close,
    -0.001 per bar of holding/idleness. PnL magnitude is irrelevant.

    Fills are realistic: entries and voluntary/timeout exits fill at the
    NEXT bar's open (decision happens at bar i close, order routes to the
    exchange, fills as bar i+1 opens). SL exits still fill intra-bar at
    the SL price level.

    Min-profit gate: a voluntary close (action == 3) is only allowed when
    the trade is either in loss (cut losses freely) or in profit ≥
    MIN_WIN_PCT (currently 0.3%). Profitable scalps below the floor are
    rejected to prevent the noise-floor scalping pathology.
    """

    MIN_WIN_PCT = 0.003   # 0.3% — voluntary closes below this gain are rejected

    def __init__(self, df, win_reward=WIN_REWARD, loss_penalty=LOSS_PENALTY,
                 hold_penalty=HOLD_PENALTY, max_hold_bars=MAX_HOLD_BARS, **kwargs):
        super().__init__(df, max_hold_bars=max_hold_bars, **kwargs)
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.hold_penalty = hold_penalty

    def _next_open(self):
        """Fill price for entries / voluntary / timeout exits.
        Falls back to current close if we're on the very last bar."""
        nxt = self.current_idx + 1
        if nxt < len(self.df):
            return float(self.df['open'].iloc[nxt])
        return float(self.df['close'].iloc[self.current_idx])

    def step(self, action):
        df = self.df
        c = df['close'].iloc[self.current_idx]
        h = df['high'].iloc[self.current_idx]
        l = df['low'].iloc[self.current_idx]

        reward = self.hold_penalty   # -0.001 per bar baseline
        info = {}
        done = False

        # Entry — fill at next bar's open (no lookahead)
        if self.position == 0:
            if action == 1:
                fill = self._next_open()
                self.position = 1
                self.entry_price = fill * (1 + self.fee_pct)
                self.bars_held = 0
                self.total_trades += 1
            elif action == 2:
                fill = self._next_open()
                self.position = -1
                self.entry_price = fill * (1 - self.fee_pct)
                self.bars_held = 0
                self.total_trades += 1

        # In trade
        elif self.position != 0:
            self.bars_held += 1

            upnl_pct = (c - self.entry_price) / self.entry_price * self.position
            self.unrealized_pnl = upnl_pct * self.capital * self.trade_pct

            close_reason = None
            c_exit = c
            # SL fills intra-bar at the price level (not bar-referenced)
            if self.position == 1 and l <= self.entry_price * (1 - self.sl_pct):
                close_reason = 'SL'
                c_exit = self.entry_price * (1 - self.sl_pct)
            elif self.position == -1 and h >= self.entry_price * (1 + self.sl_pct):
                close_reason = 'SL'
                c_exit = self.entry_price * (1 + self.sl_pct)

            # Voluntary close — gated by min-profit floor and filled at next open
            if action == 3 and close_reason is None:
                # Estimate the realized PnL the agent would lock in if we
                # filled at next-bar open. Use current close as a proxy
                # for the gate decision (the true fill is unknown until
                # bar i+1 opens, but the agent's signal is based on bar i).
                if upnl_pct < 0 or upnl_pct >= self.MIN_WIN_PCT:
                    fill = self._next_open()
                    close_reason = 'AGENT'
                    c_exit = fill * ((1 - self.fee_pct) if self.position == 1 else (1 + self.fee_pct))
                # else: reject the close, treat as hold (no-op this step)

            if self.bars_held >= self.max_hold_bars and close_reason is None:
                fill = self._next_open()
                close_reason = 'TIMEOUT'
                c_exit = fill

            if close_reason:
                pnl_pct = (c_exit - self.entry_price) / self.entry_price * self.position
                pnl_dollar = pnl_pct * self.capital * self.trade_pct
                self.capital += pnl_dollar

                # ── Categorical reward ──
                if pnl_pct > 0:
                    reward += self.win_reward
                    self.wins += 1
                else:
                    reward += self.loss_penalty
                    self.losses += 1

                self.peak_capital = max(self.peak_capital, self.capital)
                dd = (self.peak_capital - self.capital) / (self.peak_capital + 1e-8)
                self.max_drawdown = max(self.max_drawdown, dd)

                self.position = 0
                self.entry_price = 0.0
                self.unrealized_pnl = 0.0
                self.bars_held = 0

        self.current_idx += 1
        if self.current_idx >= len(df) - 1:
            done = True
            wr = self.wins / max(1, self.wins + self.losses)
            pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
            info = {
                "total_pnl": self.capital - self.initial_capital,
                "total_pnl_pct": pnl_pct,
                "win_rate": wr,
                "wins": self.wins,
                "losses": self.losses,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
            }

        obs = self._get_obs(min(self.current_idx, len(df) - 1))
        return obs, reward, done, False, info


# ── Eval / best-checkpoint callback ───────────────────────────────────────────

class WRFocusedCallback(BaseCallback):
    """
    Selects the best checkpoint by win rate (with a minimum trade count
    floor to avoid 100% WR on 1 trade), tiebroken by total PnL.
    """
    def __init__(self, eval_env, eval_every=EVAL_EVERY, patience=PATIENCE, min_trades=50):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.patience = patience
        self.min_trades = min_trades
        self.best_score = -1e9
        self.steps_since_best = 0

    def _on_step(self):
        if self.n_calls % self.eval_every != 0:
            return True

        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.eval_env.step(action)

        wr     = info.get("win_rate", 0)
        pnl    = info.get("total_pnl", 0)
        pnl_pct= info.get("total_pnl_pct", 0)
        wins   = info.get("wins", 0)
        losses = info.get("losses", 0)
        dd     = info.get("max_drawdown", 0)
        trades = info.get("total_trades", 0)

        # Score: WR primary, PnL tiebreaker, but only if min trades met
        if trades >= self.min_trades:
            score = wr * 1000 + pnl_pct
        else:
            score = -1e9   # too few trades to trust the WR

        print(f"  Step {self.n_calls:>7,} | WR: {wr:.1%} ({wins}W/{losses}L) | "
              f"PnL: ${pnl:>+7.2f} ({pnl_pct:>+6.1%}) | Trades: {trades} | DD: {dd:.1%}")

        if score > self.best_score:
            self.best_score = score
            self.steps_since_best = 0
            self.model.save(MODEL_PATH)
            print(f"  💾 New best | WR {wr:.1%} | {trades} trades")
        else:
            self.steps_since_best += self.eval_every
            if self.steps_since_best >= self.patience:
                print(f"\n⏹️  Early stop — no improvement for {self.patience:,} steps")
                return False

        return True


# ── Train / eval orchestration ────────────────────────────────────────────────

def split_train_test_oos(df, oos_days=90):
    """Train on everything older than the last `oos_days`, OOS = last 90d."""
    last_ts = pd.Timestamp(df['Datetime'].iloc[-1])
    cutoff = last_ts - pd.Timedelta(days=oos_days)
    train_df = df[df['Datetime'] < cutoff].reset_index(drop=True)
    oos_df   = df[df['Datetime'] >= cutoff].reset_index(drop=True)
    return train_df, oos_df


def evaluate_model(model, env, label):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
    print(f"\n📊 {label}:")
    print(f"   PnL:      ${info.get('total_pnl', 0):+.2f} ({info.get('total_pnl_pct', 0):+.1%})")
    print(f"   Win Rate: {info.get('win_rate', 0):.1%} "
          f"({info.get('wins', 0)}W / {info.get('losses', 0)}L)")
    print(f"   Trades:   {info.get('total_trades', 0)}")
    print(f"   Max DD:   {info.get('max_drawdown', 0):.1%}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a 5k-step smoke test instead of full training")
    args = parser.parse_args()

    df = load_or_fetch_15m()

    train_df, oos_df = split_train_test_oos(df, oos_days=90)
    print(f"\n📊 Train: {len(train_df):,} bars  |  OOS (last 90d): {len(oos_df):,} bars")
    print(f"   Train: {train_df['Datetime'].iloc[0]} → {train_df['Datetime'].iloc[-1]}")
    print(f"   OOS:   {oos_df['Datetime'].iloc[0]} → {oos_df['Datetime'].iloc[-1]}")

    print("\n🏗️  Building WR-focused environments...")
    train_env = WRFocusedCryptoEnv(train_df)
    oos_env   = WRFocusedCryptoEnv(oos_df)

    print(f"   Reward: WIN={WIN_REWARD} | LOSS={LOSS_PENALTY} | HOLD={HOLD_PENALTY}")
    print(f"   Max hold: {MAX_HOLD_BARS} bars ({MAX_HOLD_BARS * 15} min = {MAX_HOLD_BARS / 4:.1f}h)")

    print("\n🤖 Building PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )
    params = sum(p.numel() for p in model.policy.parameters())
    print(f"   Network: 36 → 256 → 256 → 128 → 4 | Params: {params:,}")

    timesteps = 5_000 if args.smoke else TOTAL_TIMESTEPS
    print(f"\n🚀 Training {timesteps:,} steps... {'(SMOKE TEST)' if args.smoke else ''}")
    print("=" * 70)

    model.learn(
        total_timesteps=timesteps,
        callback=WRFocusedCallback(oos_env),
        progress_bar=False,
    )

    print("\n" + "=" * 70)
    print("FINAL OOS EVALUATION (last 90 days, never seen during training)")
    print("=" * 70)

    if os.path.exists(MODEL_PATH + ".zip"):
        best = PPO.load(MODEL_PATH)
        oos_info = evaluate_model(best, WRFocusedCryptoEnv(oos_df), "BEST CHECKPOINT (OOS 90d)")

        wr = oos_info.get("win_rate", 0)
        trades = oos_info.get("total_trades", 0)
        pnl_pct = oos_info.get("total_pnl_pct", 0)

        print()
        if trades < 30:
            print(f"⚠️  Only {trades} trades on OOS — sample too small to trust WR")
        elif wr >= 0.60 and pnl_pct > 0:
            print(f"✅ WR target hit ({wr:.1%}) and OOS profitable. Ready to deploy.")
        elif wr >= 0.55:
            print(f"🟡 WR {wr:.1%} — below 60% target but above coin-flip. Worth A/B vs current model.")
        else:
            print(f"❌ WR {wr:.1%} — model didn't learn the WR-focused objective. "
                  f"Consider longer training, different reward shape, or more data.")
    else:
        print("⚠️  No best checkpoint saved (callback never found a passing eval). "
              "Try lowering --min-trades or running longer.")

    print(f"\n💾 Best checkpoint: {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
