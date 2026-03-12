# Phase 4: Exit Redesign — Tailor Stop/TP/Trailing Logic to Active Regime

## Context
Part of the 6-phase roadmap from `memory/improvement_suggestions_checkpoint.md`.
Phases 1, 2, and 3 must be complete before starting this phase.

## Goal
The current exit logic is regime-blind: same ATR stop and ATR TP multipliers are used regardless
of whether the market is trending, ranging, or transitioning. Regime-specific exits should:
- **Trend regime**: wide stop (ride the move), large TP or trailing exit when trend weakens
- **Range regime**: tight stop (mean-reversion trade), fast TP capture before next reversal

## Problem Statement
Current code:
1. `risk.atr_stop_mult` (2.75) and `risk.atr_tp_mult` (15.0) are single global values
2. Engine applies the same SL/TP regardless of what regime was active at entry
3. `signal_generator.py:compute_stops()` uses a single multiplier for all trades
4. No trailing stop logic exists — trend trades ride to TP or timeout

## Changes

### 1. Regime-specific stop/TP multipliers in base.yaml
**File**: `code3.0/configs/base.yaml`

Replace single `risk.atr_stop_mult` / `risk.atr_tp_mult` with regime-specific values:
```yaml
risk:
  # trend regime (bull or bear)
  trend_atr_stop_mult:  2.75
  trend_atr_tp_mult:    12.0
  # range/sideways regime
  range_atr_stop_mult:  1.5
  range_atr_tp_mult:    3.0
  # keep existing keys as fallback labels (set equal to trend values)
  atr_stop_mult:  2.75   # used by walk-forward + optimizer (1H mode)
  atr_tp_mult:    12.0
```

### 2. Compute regime-aware stops in generator.py
**File**: `code3.0/src/trade2/signals/generator.py`

Read `compute_stops()` function. Modify to accept an optional `regime_labels` Series.
When regime_labels is provided:
- For bars labelled "bull" or "bear": use `trend_atr_stop_mult` / `trend_atr_tp_mult`
- For bars labelled "sideways" or any other: use `range_atr_stop_mult` / `range_atr_tp_mult`

Signature change:
```python
def compute_stops(df, atr_stop_mult, atr_tp_mult, config=None, regime_labels=None):
```

When `config` and `regime_labels` are provided, override per-bar with regime-specific multipliers.
Otherwise fall back to the scalar `atr_stop_mult` / `atr_tp_mult` passed in (backward compat).

### 3. Wire regime labels into compute_stops call in run_pipeline.py
**File**: `code3.0/src/trade2/app/run_pipeline.py`

Find the `compute_stops()` calls for train/val/test. Pass:
- `config=config`
- `regime_labels` = the forward-filled regime series aligned to the signal df index

For multi_tf mode, regime labels are on 1H but signals are on 5M — use the forward-filled
`hmm_regime` column already present in `train_sig_df` (set by `forward_fill_1h_regime()`).
For single_tf mode, use `train_labels` directly.

### 4. Trailing stop for trend trades (optional, config-gated)
**File**: `code3.0/src/trade2/backtesting/engine.py`

Add optional trailing stop logic. Activated by new config key:
```yaml
risk:
  use_trailing_stop: false   # start false, enable for testing
  trailing_atr_mult: 1.5    # trail by N * ATR from highest favorable excursion
```

When `use_trailing_stop: true` and direction == "long":
- Track `max_favorable_close` since entry
- If `close[i] < max_favorable_close - trailing_atr * atr[i]`: trigger trailing exit
Requires passing `atr` array into `_simulate_trades()`. Add as optional param `atr: np.ndarray = None`.

### 5. Add trailing stop params to base.yaml
Already covered above. Add under `risk`:
```yaml
  use_trailing_stop: false
  trailing_atr_mult: 1.5
```

## Files to Modify
1. `code3.0/configs/base.yaml` — regime-specific stop/TP multipliers + trailing stop params
2. `code3.0/src/trade2/signals/generator.py` — regime-aware compute_stops()
3. `code3.0/src/trade2/app/run_pipeline.py` — pass regime_labels to compute_stops()
4. `code3.0/src/trade2/backtesting/engine.py` — optional trailing stop logic

## Verification
1. `trade2 --retrain-model --skip-walk-forward`
2. Print stop_long/tp_long min/max — should see two distinct clusters (tight for range, wide for trend)
3. Check exit_reason breakdown: regime-specific TP hit rates should differ by regime
4. Compare test metrics vs Phase 3 baseline
5. Toggle `use_trailing_stop: true` and rerun — check if trend trades improve

## Before Starting
- Read all 4 files before making changes
- Check memory/MEMORY.md for Phase 3 baseline results
- Read `forward_fill_1h_regime()` in signals/regime.py to confirm `hmm_regime` column name
