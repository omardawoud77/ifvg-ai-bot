"""
Phase 2 — Smoke-test training run
=================================
END-TO-END pipeline proof, NOT a deployable model:

    data → crypto_env_v3 → PPO train → walkforward eval → verdict

One symbol (BTC), one fold (~10.5 months sliced so 8m train / 2m test),
a deliberately short PPO run (default 120k steps ≈ 15-30 min on M4 CPU).
A PASS here would be luck; the point is that nothing crashes and every
interface (env obs, exits, fees, fold metrics, verdict) lines up before
the real July-18 training.

    python smoke_train.py                 # default 120k steps
    python smoke_train.py --steps 30000   # faster, pure plumbing check
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from crypto_env_v3 import CryptoEnvV3, build_mtf_dataset, build_feature_matrix
from walkforward import FoldSpec, walk_forward

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "smoke")   # git-ignored


def _env_from_slice(df_slice: pd.DataFrame, feats_slice: pd.DataFrame,
                    episode_len: int, seed: int | None = 7) -> CryptoEnvV3:
    df = df_slice.reset_index(drop=True)
    mat = np.clip(feats_slice.values.astype(np.float32), -10.0, 10.0)
    return CryptoEnvV3(df, mat, episode_len=episode_len, seed=seed)


def make_train_fn(steps: int):
    def train_fn(train_df: pd.DataFrame, train_feats: pd.DataFrame):
        from stable_baselines3 import PPO
        env = _env_from_slice(train_df, train_feats, episode_len=720)
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256,
                    seed=7, verbose=0, device="cpu")
        t0 = time.time()
        model.learn(total_timesteps=steps, progress_bar=False)
        print(f"    trained {steps} steps in {time.time() - t0:.0f}s")
        return model
    return train_fn


def eval_fn(model, test_df: pd.DataFrame, test_feats: pd.DataFrame) -> pd.DataFrame:
    """Deterministic policy rollout across the whole test window.
    Same fees as training env (taker both sides) per walkforward.FEE_NOTE."""
    env = _env_from_slice(test_df, test_feats,
                          episode_len=len(test_df) - 32, seed=None)
    obs, _ = env.reset(options={"start_idx": 30})
    rows, side, done = [], 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        if env.position != 0:
            side = env.position
        if info.get("trade_closed"):
            rows.append({"pnl_pct": info["pnl_pct"],
                         "side": "LONG" if side == 1 else "SHORT",
                         "close_reason": info["close_reason"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    print(f"[smoke] building dataset + features for {args.symbol}...")
    df = build_mtf_dataset(args.symbol)
    feats_mat = build_feature_matrix(args.symbol, df)
    feats = pd.DataFrame(feats_mat)

    # slice to the most recent ~10.5 months → exactly one 8m/2m fold
    ts = pd.to_datetime(df["open_time"], utc=True,
                        format="ISO8601", errors="coerce")
    cutoff = ts.max() - pd.DateOffset(months=10, days=15)
    keep = (ts >= cutoff).values
    df, feats = df[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)
    print(f"[smoke] window: {ts[keep].min():%Y-%m-%d} → {ts.max():%Y-%m-%d} "
          f"({keep.sum()} bars, expecting 1 fold)")

    report = walk_forward(df, feats, make_train_fn(args.steps), eval_fn,
                          symbol=f"{args.symbol}-SMOKE",
                          spec=FoldSpec(train_months=8, test_months=2,
                                        step_months=2))
    print()
    print(report.summary())
    os.makedirs(OUT_DIR, exist_ok=True)
    report.to_csv(os.path.join(OUT_DIR, "smoke_report.csv"))
    print(f"\n[smoke] report saved. Pipeline verified end-to-end "
          f"(verdict above is a {args.steps}-step toy — not a deployment call).")


if __name__ == "__main__":
    main()
