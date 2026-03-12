# Phase 5: Systems Speed — Caching, Loop Optimization, Reduce Per-Trial Recomputation

## Context
Part of the 6-phase roadmap from `memory/improvement_suggestions_checkpoint.md`.
Phases 1-4 must be complete. This phase does NOT change strategy logic — only speed.

## Goal
Make the pipeline fast enough to run 500+ Optuna trials in a reasonable time.
Current bottleneck: every Optuna trial reloads data, recomputes all features, and reruns
the full HMM fit. Most of this work is redundant across trials with the same data split.

## Profiling First
Before making any changes, profile the pipeline:
```python
# Add to run_pipeline.py temporarily or run as standalone:
import cProfile, pstats
cProfile.run("run_pipeline(config, skip_walk_forward=True)", "profile_output")
stats = pstats.Stats("profile_output")
stats.sort_stats("cumulative").print_stats(20)
```
Identify the top 3 bottlenecks by cumulative time. Only optimize what profiling confirms is slow.

## Expected Bottlenecks (confirm with profiling)
1. **HMM fit**: `XAUUSDRegimeModel.fit()` retrains on every walk-forward window
2. **Feature engineering**: `add_1h_features()` and `add_5m_features()` re-run on every Optuna trial
3. **Backtest inner loop**: `_simulate_trades()` pure Python loop over 400k+ bars

## Changes

### 1. Cache feature DataFrames in optimizer
**File**: `code3.0/src/trade2/optimization/optimizer.py`

Read the file first. The Optuna objective function likely rebuilds features on every trial.
Refactor: compute features ONCE before the trial loop, pass pre-built DataFrames into the objective.
The optimizer should only re-run `generate_signals()`, `compute_stops()`, and `run_backtest()` per trial.

Pattern:
```python
# Outside objective (run once):
train_feat = add_1h_features(train_raw, config)  # cache this
val_feat   = add_1h_features(val_raw, config)

# Inside objective (per-trial):
signals = generate_signals(val_feat, config, ...)
stops   = compute_stops(signals, ...)
metrics, _ = run_backtest(stops, ...)
```

### 2. Numba/vectorized inner loop for backtest engine
**File**: `code3.0/src/trade2/backtesting/engine.py`

The `_simulate_trades()` function is a pure Python loop — the main speed bottleneck for 5M data
(424k bars × N trials).

Option A (preferred, no new deps): Vectorize the SL/TP check using numpy operations.
The bar-by-bar state machine is hard to fully vectorize, but the SL/TP hit detection can be:
```python
# Pre-compute which bars hit SL or TP using array ops, then resolve order conflicts
sl_hit = (lows <= frozen_sl)  # array
tp_hit = (highs >= frozen_tp) # array
first_exit = np.argmax(sl_hit | tp_hit)  # fast
```
This is only valid for single-position segments. Implement as a helper for each trade segment.

Option B (fallback): Add `@numba.njit` decorator if numba is available.
Check with `try: import numba` first. If available, extract the core loop into a numba-jittable function.

### 3. HMM fit caching in walk-forward
**File**: `code3.0/src/trade2/backtesting/engine.py` (run_walk_forward)

Each walk-forward window retrains the HMM from scratch. This is correct — HMM must be retrained
per window. But features can still be cached:
- Compute `train_feat` and `val_feat` once per window outside the try block
- Only retry HMM fit on exception

### 4. Add pipeline timing log
**File**: `code3.0/src/trade2/app/run_pipeline.py`

Wrap each major pipeline stage with a timer:
```python
import time
t0 = time.perf_counter()
# ... stage ...
print(f"[pipeline] {stage_name}: {time.perf_counter()-t0:.1f}s")
```
Stages to time: data load, feature engineering, HMM fit, signal gen, each backtest, walk-forward.
This makes future bottleneck identification easy without a full profiler.

### 5. Config: optimization trial count
No changes to base.yaml — speed improvements should let users increase `optimization.n_trials`
to 500+ without changing the config structure.

## Files to Modify
1. `code3.0/src/trade2/optimization/optimizer.py` — cache features outside trial loop
2. `code3.0/src/trade2/backtesting/engine.py` — vectorize SL/TP detection, time walk-forward
3. `code3.0/src/trade2/app/run_pipeline.py` — add timing log per stage

## Verification
1. Run pipeline twice: measure wall-clock time before and after changes
2. Run `trade2 --optimize --trials 50` — confirm no errors, faster per-trial time
3. Confirm strategy output (metrics) is identical before/after (speed changes must not alter results)
4. Check timing log output shows where time is spent

## Before Starting
- Profile first — do NOT optimize blindly
- Read optimizer.py carefully before changing it (it was recently modified per git status)
- Do not change any strategy logic — this phase is speed only
- Check memory/MEMORY.md for Phase 4 baseline results
