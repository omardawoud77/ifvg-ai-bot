# Phase 2 Scaffolding — Retraining Infrastructure

Built 2026-07-05. Runs entirely on your Mac. **Nothing here touches Railway
or the live bot.** The worker only ever receives a finished model `.zip`.

## What's here

| File | Purpose | Status |
|---|---|---|
| `data_pipeline.py` | Downloads 3y of 1H/4H/1D klines + funding history for BTC/ETH/SOL from Binance Futures public REST, caches to `./data/` | ready to run |
| `features_v3.py` | The 14 new observation features: market structure (swing pivots, HH/HL state), prior-day levels, funding rate (raw / z-score / 3d cum), BTC-lead (returns, rolling corr, relative strength) | tested on synthetic data, causality-verified |
| `walkforward.py` | Rolling train/OOS-test harness with the hard deployment criterion (pooled OOS expectancy > 0, ≥30 trades, no fold with >15% drawdown) | self-test passes; correctly FAILS a coin-flip strategy |

## Order of operations

```
1. pip install pandas numpy requests
2. python data_pipeline.py            # ~10-20 min, one-time download
3. python features_v3.py --check      # sanity stats on the real data
4. (wait for shadow data → spec)      # ~July 18 calibration session
5. crypto_env_v3 + train.py           # built after spec, using pieces above
6. python walkforward.py drives PPO train/eval per symbol
7. Deploy ONLY on a PASS — copy .zip to repo, point live agent at it
```

## Integration points (deliberately left open)

- **`crypto_env_v3`** — needs the CURRENT env file to be a drop-in match
  (observation layout, action space, reward, episode mechanics). The v3 env
  will concatenate `[existing_44_features, features_v3]` → 58-dim obs, and
  replicate LIVE exit behavior exactly: tiered SL (2.5–3%), 5% TP,
  breakeven at +2.5%, 48h hold limit, real taker fees. **Send me the
  current env .py file and this gets built in one pass.**
- **`train_fn` / `eval_fn`** — the two callables `walkforward.walk_forward()`
  takes. `train_fn` wraps `PPO(...).learn()`; `eval_fn` runs the trained
  policy through the test window (through the SAME gate logic as live, or
  raw — we'll do both, so we can measure the gate's contribution) and
  returns one row per simulated trade, net of fees.

## Decisions locked in by design

- **Balanced regime sampling** happens at the episode-sampling level inside
  env v3 (train episodes drawn proportionally from bull/bear/sideways
  segments, labeled by 200-EMA slope on 1D), not by distorting the data.
- **Walk-forward, not single split.** 8-month train / 2-month test, rolling
  by 2 → every OOS month is tested exactly once, no cherry-picking.
- **No re-roll rule.** If a run fails deployment, we change the spec and
  re-run — we do not re-seed until green.

## Still blocked on shadow data (~July 18)

The spec decisions the calibration analysis will make:
- which conditions the current model is miscalibrated in (target features)
- whether regime becomes a model INPUT vs stays a gate condition
- whether reward shaping needs asymmetry (the P(SHORT)≈1–2% problem may be
  data imbalance, reward design, or both)
- final threshold placement (0.55/0.45) informed by realized calibration
