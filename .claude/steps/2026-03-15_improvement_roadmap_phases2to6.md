# Plan: Improvement Roadmap Phases 2–6

**Date planned:** 2026-03-15
**Motivation:** Complete the 6-phase improvement roadmap from code3_improvement_suggestions.pdf. Phase 1 (correctness) was done on 2026-03-12.

## Phases

### Phase 2 — Strategy Simplification
- `enabled` flag on all strategies (trend/range/volatile/cdc) in router + config
- `require_bos_confirm` config key on trend strategy (default false)

### Phase 3 — Probabilistic HMM
- Already completed 2026-03-12 (prob-only gating, min_confidence=0.45, cooldown=2h, linear sizing)

### Phase 4 — Exit Redesign (Trailing Stops)
- `engine.py`: trailing stop logic in `_simulate_trades()` — updates `frozen_sl` each bar
- `generator.py`: `compute_stops_regime_aware()` writes `trailing_atr_mult_long/short` columns
- Config: `trailing_enabled: false`, `trailing_atr_mult: 1.5` added to trend strategy

### Phase 5 — Systems Speed (Feature Caching)
- `run_pipeline.py`: `_load_features_cached()` wraps feature build calls with pickle cache
- Config: `pipeline.cache_features: false` (set to true to activate)

### Phase 6 — Hard Validation (Stricter Cross-Split Checks)
- `hard_rejection.py`: 3 new rules using `train_metrics` + `val_metrics` params:
  - Rule 5: train split `min_trades_train` (default 30)
  - Rule 6: val split `min_trades_val` (default 10)
  - Rule 7: test/train Sharpe ratio > `max_overfitting_sharpe_ratio` (default 3.0)
- `run_pipeline.py`: passes `train_metrics` + `val_metrics` to `hard_rejection_checks()`

## Files Modified
- `code3.0/src/trade2/signals/strategies/trend.py`
- `code3.0/src/trade2/backtesting/engine.py`
- `code3.0/src/trade2/signals/generator.py`
- `code3.0/src/trade2/evaluation/hard_rejection.py`
- `code3.0/src/trade2/app/run_pipeline.py`
- `code3.0/configs/base.yaml`

## Completed
**Date:** 2026-03-15
**Commit:** 904d704
