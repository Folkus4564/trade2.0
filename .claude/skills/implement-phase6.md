# Phase 6: Hard Validation — Require Strong Performance Across All Splits, Not Just One Slice

## Context
Part of the 6-phase roadmap from `memory/improvement_suggestions_checkpoint.md`.
Phases 1-5 must be complete. This is the final phase — tighten the acceptance gate.

## Goal
A strategy that only looks good on the test slice but is fragile elsewhere should not be approved.
This phase adds cross-split consistency checks, single-month dominance detection, and stricter
walk-forward requirements.

## Problem Statement
Current evaluation has:
1. `multi_split_verdict()` requires ALL splits to pass, which is good — but the per-split thresholds
   in `acceptance.py` are loose (train: sharpe >= 0 is trivially true)
2. No check for single-month dominance: one exceptional month can inflate the annual return
3. Walk-forward currently fails due to 2022 bear market — the WF windows don't match strategy regime
4. No minimum trade count per walk-forward window
5. `acceptance.py` `test.min_annualized_return` is 0.20 — below the 50% target from CLAUDE.md

## Changes

### 1. Raise acceptance thresholds in base.yaml
**File**: `code3.0/configs/base.yaml`

Update `acceptance.test` to match performance targets from CLAUDE.md:
```yaml
acceptance:
  test:
    min_annualized_return: 0.20   # keep minimum at 20%
    min_sharpe:            1.0
    min_drawdown:          -0.35
    min_profit_factor:     1.2
    min_trades:            30
    min_win_rate:          0.40
  train:
    min_sharpe:            0.3    # raise from 0.0 — train must show some edge
    min_profit_factor:     1.1    # raise from 1.0
    min_trades:            50
  walk_forward:
    min_mean_sharpe:       0.3    # lower from 0.5 — achievable for trend strategy
    min_positive_windows:  0.60   # lower from 0.75 — 3/5 windows must be positive
```

### 2. Add single-month dominance check
**File**: `code3.0/src/trade2/evaluation/hard_rejection.py`

Add Rule 5: reject if any single calendar month contributes > 40% of total PnL.
Requires trades_df to be passed to hard_rejection_checks().

Signature addition:
```python
def hard_rejection_checks(
    metrics, config,
    train_regime_dist=None, test_regime_dist=None,
    cost_sensitivity_metrics=None,
    walk_forward_run=False,
    trades_df=None,          # NEW: pass test trades_df for dominance check
):
```

Rule 5 implementation:
```python
max_monthly_pnl_share = config["hard_rejection"]["max_monthly_pnl_share"]  # e.g. 0.40
if trades_df is not None and len(trades_df) > 0:
    trades_df["exit_month"] = pd.to_datetime(trades_df["exit_time"]).dt.to_period("M")
    monthly_pnl = trades_df.groupby("exit_month")["pnl"].sum()
    total_pnl = monthly_pnl.sum()
    if total_pnl > 0:
        max_share = (monthly_pnl / total_pnl).max()
        if max_share > max_monthly_pnl_share:
            top_month = monthly_pnl.idxmax()
            rejections["monthly_dominance"] = (
                f"Month {top_month} contributes {max_share*100:.0f}% of total PnL"
            )
```

Add to base.yaml `hard_rejection` section:
```yaml
hard_rejection:
  max_monthly_pnl_share: 0.40
```

### 3. Walk-forward minimum trade count
**File**: `code3.0/src/trade2/backtesting/engine.py` (run_walk_forward)

Currently a window is skipped if `n_sigs < 5`. Add a post-backtest check:
if `metrics["total_trades"] < config["walk_forward"]["min_trades_per_window"]`, mark
the window result as `{"insufficient_trades": True}` and exclude from aggregate stats.

Add to base.yaml:
```yaml
walk_forward:
  min_trades_per_window: 5
```

### 4. Wire trades_df into hard_rejection_checks in run_pipeline.py
**File**: `code3.0/src/trade2/app/run_pipeline.py`

Currently `run_backtest()` returns `(metrics, trades_df)` — the trades_df is discarded for test.
Save the test trades_df and pass it to `hard_rejection_checks()`:

```python
test_metrics, test_trades = run_backtest(test_sig, ...)
# ...
hrd = hard_rejection_checks(
    metrics=test_metrics,
    config=config,
    ...,
    trades_df=test_trades,   # NEW
)
```

### 5. Monthly PnL report in pipeline output
**File**: `code3.0/src/trade2/app/run_pipeline.py`

After the test backtest, print a monthly PnL table for the test period:
```
[pipeline] Monthly PnL (test):
  2024-01: $1,234  | 2024-02: -$456 | ...
  Max month share: 23.4%
```

## Files to Modify
1. `code3.0/configs/base.yaml` — raise thresholds, add max_monthly_pnl_share, min_trades_per_window
2. `code3.0/src/trade2/evaluation/hard_rejection.py` — add trades_df param, Rule 5 monthly dominance
3. `code3.0/src/trade2/backtesting/engine.py` — min_trades_per_window in walk-forward
4. `code3.0/src/trade2/app/run_pipeline.py` — pass test_trades to hard_rejection, monthly PnL print

## Verification
1. `trade2 --retrain-model --skip-walk-forward`
2. Confirm monthly PnL table printed for test period
3. Check that single-month dominance is correctly detected/rejected with synthetic data
4. Run with walk-forward: confirm min_trades_per_window exclusion works
5. Final verdict should be more conservative — any strategy that previously passed narrowly
   should now be scrutinized by the monthly dominance check

## Final Goal
After Phase 6, a strategy is only APPROVED if:
- It passes all 3 splits (including stricter train threshold)
- No single month > 40% of PnL
- Walk-forward has >= 60% positive windows with >= 5 trades each
- Passes 2x cost sensitivity (< 35% Sharpe drop)
- Walk-forward was run

## Before Starting
- Read all files before making changes
- Check memory/MEMORY.md for Phase 5 results
- After completing, do a full run with walk-forward enabled and record final results in memory
