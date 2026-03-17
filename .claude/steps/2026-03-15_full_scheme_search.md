# Plan: `full_scheme_search` - Brute Force 50% Return Search

## Context

Current best result is ~12.86% annualized return (Sharpe 1.179) from legacy signal generation with 1H regime + 5M signals. The regime-specialized router gives 0.391 Sharpe due to architectural exit merge issues. Target: **50% annualized return with max drawdown -25%**.

The `_empty_signals()` exit bug is already fixed (exit_long=1 for disabled strategies). The router exit merge and param alignment are still needed but are separate from this search effort.

## User Decisions

- **No early stopping**: Run all 20 ideas to completion, pick overall best
- **HMM features**: Config-driven (make `hmm.features` in YAML actually control which features HMM uses)
- **Max drawdown**: -25% hard cap — filter out any result exceeding this
- **Optuna target**: Run each idea TWICE (once optimizing val_sharpe, once val_return) = **40 total experiments**
- **Runtime estimate**: ~60-120 minutes total

## Approach

Create a `full_scheme_search` CLI command that:
1. Defines 20 experiment "ideas" (12 single-dimension + 8 combos)
2. For each idea: runs pipeline TWICE (Optuna targeting val_sharpe, then val_return) = 40 experiments
3. Filters results by max_drawdown >= -25%
4. Ranks all passing experiments by test annualized return
5. Re-runs top 3 with walk-forward validation

Uses **legacy signal mode** (proven Sharpe 1.179) as the base.

## The 20 Ideas

### Single-dimension experiments (1-12)

| # | Idea | Key Changes |
|---|------|-------------|
| 1 | **Baseline + Optuna** | Legacy mode, 100 Optuna trials |
| 2 | **4H regime + 5M signals** | regime_timeframe=4H |
| 3 | **1H regime + 15M signals** | signal_timeframe=15M |
| 4 | **1H regime + 30M signals** | signal_timeframe=30M |
| 5 | **4H regime + 15M signals** | Both slower TFs |
| 6 | **HMM 2-state** | n_states=2 |
| 7 | **HMM 4-state** | n_states=4 |
| 8 | **Aggressive sizing** | base_alloc=0.90, sizing_max=2.0 |
| 9 | **No session filter** | session.enabled=false |
| 10 | **Wide stops, ride trends** | atr_stop=4.0, atr_tp=25.0 |
| 11 | **Quick scalp** | atr_stop=1.5, atr_tp=3.0, max_hold=24 |
| 12 | **HMM minimal features** | Only ret+atr+vol (3 features) |

### Combo experiments (13-20)

| # | Idea | Key Changes |
|---|------|-------------|
| 13 | **4H + aggressive + wide stops** | 4H regime, alloc=0.90, sizing=2.0, stop=4.0, tp=25.0 |
| 14 | **4H + 15M + no session** | 4H regime, 15M signals, session off |
| 15 | **2-state + aggressive + wide** | HMM 2-state, alloc=0.90, sizing=2.0, stop=4.0 |
| 16 | **4H + 2-state + aggressive** | 4H regime, 2-state HMM, alloc=0.90, sizing=2.0 |
| 17 | **15M + no session + aggressive** | 15M signals, session off, alloc=0.90, sizing=2.0 |
| 18 | **30M + 4-state + wide stops** | 30M signals, 4-state HMM, stop=4.0, tp=25.0 |
| 19 | **4H + minimal HMM + aggressive** | 4H regime, 3 HMM features, alloc=0.90, sizing=2.0 |
| 20 | **Kitchen sink** | 4H regime, 15M signals, 2-state HMM, no session, alloc=0.90, sizing=2.0, stop=4.0, tp=25.0 |

## Files Changed

### New Files
- `code3.0/src/trade2/app/full_scheme_search.py` - Main batch runner (20 ideas x 2 targets)

### Modified Files
- `code3.0/src/trade2/app/run_pipeline.py` - TF generalization + optuna_target param + legacy_signals→optimizer consistency fix
- `code3.0/src/trade2/features/hmm_features.py` - Config-driven feature selection via config["hmm"]["features"]
- `code3.0/src/trade2/signals/generator.py` - Added `_get_bars_per_hour` helper for 15M/30M support
- `code3.0/src/trade2/backtesting/engine.py` - Multi-TF freq support (15min, 30min)
- `code3.0/src/trade2/optimization/optimizer.py` - optuna_target param + config-based tf_scale
- `code3.0/src/trade2/signals/router.py` - Exit merge: AND → OR, disabled returns 0
- `code3.0/configs/base.yaml` - Trend params aligned (min_prob 0.70→0.77, adx 20→15, stop 2.5→2.75)
- `code3.0/pyproject.toml` - Added full_scheme_search CLI entry point

## Usage

```bash
# Run full 40-experiment search
full_scheme_search

# Quick test with fewer trials
full_scheme_search --trials 20

# Run specific ideas only
full_scheme_search --ideas 2,5,13 --trials 100

# Skip walk-forward re-run
full_scheme_search --top-wf 0
```

## Completed

- **Date**: 2026-03-15
- **Commit**: 8159655
- **Status**: Implementation complete, smoke test passed (idea 1 runs correctly, both optuna targets work, leaderboard printed, results saved to artefacts/)
