# Strategy Debugger Audit Report
## XAUUSD SMC + HMM Regime Strategy

**Audit Date:** 2026-03-09
**Auditor:** Strategy Debugger Agent
**Strategy:** xauusd_smc_hmm_regime
**Overall Status:** CONDITIONAL PASS

---

## Executive Summary

The strategy passes the majority of bias checks. No critical lookahead bias or data leakage issues were found in the primary signal path. However, several warnings require attention before the strategy can be considered fully clean. The test-period metrics are within normal ranges and do not exhibit suspicious flag patterns.

---

## Critical Issues

| # | Issue | File | Location | Severity |
|---|-------|------|----------|----------|
| - | No critical issues found | -- | -- | -- |

---

## Warnings

| # | Warning | File | Location | Description |
|---|---------|------|----------|-------------|
| W1 | FVG detection reads current-bar high/low before shift | `features.py` | Lines 157-196 | The FVG loop reads `high_arr[i]` and `low_arr[i]` (bar i, the current bar) to detect a new FVG at bar i and stores the zone in `bull_fvg_lo[i]` / `bear_fvg_lo[i]`. The final `.shift(1)` on lines 211-212 does push the retest signal one bar forward, making the final output lag-safe. However the carry-forward invalidation check on line 179 (`close_arr[i] < active_bull_fvg_lo`) references bar i's close to decide whether to mark bar i's zone -- this is internally consistent since bar i is already a closed bar. The design is technically safe due to the final shift, but fragile: removing the shift would introduce lookahead. |
| W2 | Spread (3 pips) declared but not applied in backtest | `engine.py` | Lines 17-18, 85-97 | `SPREAD_PIPS = 3` is declared as a constant and stored in the results JSON, but it is NOT passed to `vbt.Portfolio.from_signals()`. Only `fees = COMMISSION_RT / 2` (1 bps per side) and `slippage` (1 pip converted to price ratio) are passed. Spread is effectively not applied as a separate cost component, meaning actual transaction costs are underestimated by approximately 3 pips per trade. Reported performance is therefore slightly optimistic. |
| W3 | Slippage calculation uses period-average price, not bar-level price | `engine.py` | Lines 76-78 | `avg_price = close.mean()` is calculated once for the entire test period. A single static slippage ratio is then passed to vectorbt for all trades. XAUUSD ranged from approximately $2,000 to $2,700 during the test period -- a fixed average-based slippage ratio underestimates costs at lower prices and overestimates at higher prices. Error magnitude is within 15% of the slippage cost, which is small in dollar terms but worth noting. |
| W4 | Walk-forward validation does not re-train HMM per window | `engine.py` / `pipeline.py` | `run_walk_forward` L149-188; `pipeline.py` L162 | The `run_walk_forward` function receives a signal DataFrame with signals already generated using the full-period HMM. It slices into windows and backtests each slice, but does NOT re-train the HMM on each window's own training data. Walk-forward results therefore test backtest stability, not true out-of-sample regime detection robustness. |
| W5 | `avg_trade_duration_bars` is NaN in test results | `xauusd_smc_hmm_regime_test_results.json` | Line 22 | Trade duration is NaN, indicating the `Duration` column was not found or was empty in `pf.trades.records_readable`. This is a vectorbt column name mismatch, not indicative of bias, but it limits full audit of execution realism and trade holding period analysis. |

---

## Passed Checks

### Lookahead Bias

- **Order Block (OB) detection loop**: PASS. Scans bars up to `i-1` when checking impulse moves (`features.py` L85-113). The impulse slice `low_arr[i-IMPULSE_BARS:i]` correctly excludes bar i (Python end-exclusive). Final `.shift(1)` on lines 128-129 ensures the OB retest flag at bar i reflects only data from bar i-1 and earlier.
- **Fair Value Gap (FVG) shift applied**: PASS. Lines 211-212 apply `.shift(1)` to `fvg_bullish` and `fvg_bearish` before storing in the output DataFrame. Despite the borderline internal design (W1), the final output is lag-safe.
- **Liquidity Sweep rolling windows**: PASS. Lines 223-224 use `high.rolling(SWING_LOOKBACK).max().shift(1)` and `low.rolling(SWING_LOOKBACK).min().shift(1)`. Sweep signals are also shifted by 1 bar on lines 241-242.
- **HMM input features shifted**: PASS. Lines 367-370 in `features.py` create `hmm_feat_ret`, `hmm_feat_rsi`, `hmm_feat_atr`, `hmm_feat_vol` all with `.shift(1)`. The HMM never receives the current bar's data when predicting the current bar's regime.
- **Donchian Channel breakout**: PASS. Lines 349-350 compute `dc_upper` and `dc_lower` with `.shift(1)`. Breakout signals compare current close against the previous completed bar's N-bar channel.
- **Log returns**: PASS. `log_ret = np.log(close / close.shift(1))` is a standard backward-looking return calculation.

### Data Leakage

- **Scaler fitted on train only**: PASS. `hmm_model.py` line 57 calls `self.scaler.fit_transform(X)` only inside `fit()`. Val and test predictions use `self.scaler.transform(X)` (lines 106, 113), which uses train-only scaler parameters.
- **HMM trained on train split only**: PASS. `pipeline.py` lines 116-121 call `get_hmm_feature_matrix(train_feat)` then `model.fit(X_train)`. Val and test features are only passed to `.predict()` / `.predict_proba()` -- no re-fitting on non-train data.
- **Test set excluded from optimization**: PASS. The `optimize()` function in `pipeline.py` line 274 discards the test split (`_`). The Optuna objective operates exclusively on `val_raw`.
- **Train/val/test split boundaries correct and non-overlapping**: PASS. `loader.py` lines 101-103 use strict boundary conditions: `train <= TRAIN_END`, `val in (TRAIN_END, VAL_END]`, `test > VAL_END`. No overlap possible.
- **Stale model deleted before pipeline**: PASS. `pipeline.py` lines 42-45 delete any pre-existing HMM pickle before running, ensuring the model is retrained fresh on the current train split every run.
- **`_map_states` uses train data only**: PASS. `_map_states` in `hmm_model.py` lines 71-98 is called from `fit()` (line 67) using the same train `X` array passed to `fit()`. It does not access val or test data.
- **Features computed independently per split**: PASS. Each split has `add_features()` called independently on lines 102-107 of `pipeline.py`. Val and test features do not incorporate statistics from the train window.

### Execution Costs

- **Commission applied**: PASS. `fees = COMMISSION_RT / 2` (1 bps per side = 2 bps round-trip) is passed to `vbt.Portfolio.from_signals()` (engine.py line 92).
- **Slippage applied**: PASS. A slippage ratio is calculated and passed to `from_signals()` (engine.py line 93).
- **Spread declared and stored**: CONDITIONAL. `SPREAD_PIPS = 3` is declared and stored in the result JSON (costs section), but not applied as a separate deduction in the backtest call (see Warning W2).
- **Cost constants present in results JSON**: PASS. `spread_pips: 3`, `slippage_pips: 1`, `commission_rt: 0.0002` are all recorded in the results file.

### Signal Logic

- **Simultaneous long/short conflict resolution**: PASS. `engine.py` lines 59-61 suppress long entries when both long and short signals fire simultaneously.
- **HMM label/probability alignment**: PASS. `signal_generator.py` lines 64-70 use `.reindex(out.index)` to align HMM outputs to the full DataFrame index, filling gaps with "sideways" and 0.0.
- **Exit logic consistency**: PASS. Exits fire when regime is no longer bull (for longs) or no longer bear (for shorts) -- lines 113-114. With the 2-state HMM (bull/bear), this is a clean regime-flip exit with no ambiguity.

### Resampling

- **OHLCV resample uses `label='left', closed='left'`**: PASS. `loader.py` line 82 prevents future-bar data from being included in any resampled OHLC bars.

---

## Performance Sanity Check

| Metric | Reported Value | Suspicious Threshold | Status |
|--------|---------------|---------------------|--------|
| Annualized Return | 19.49% | > 200%: likely bug | NORMAL |
| Sharpe Ratio | 1.2455 | > 5.0: likely bug | NORMAL |
| Max Drawdown | -7.56% | < -2% with large return: suspicious | NORMAL |
| Win Rate | 49.43% | > 80%: likely lookahead | NORMAL |
| Total Trades | 87 | < 30: insufficient | NORMAL (87 trades) |
| Profit Factor | 1.5615 | > 10.0: likely bug | NORMAL |
| Avg Win/Loss Ratio | 1.5979 | > 20.0: likely bug | NORMAL |
| Benchmark (Buy and Hold) | 27.74% | -- | NOTE: strategy underperforms B&H |
| Alpha vs Benchmark | -8.25% | -- | NEGATIVE (expected in strong trend year) |
| Information Ratio | -0.5779 | -- | NEGATIVE (note) |

**Benchmark context:** The strategy underperforms buy-and-hold XAUUSD in the test period (2024). Gold had an exceptionally strong trending year in 2024 (approximately +27.7% annualized). A regime-filtered strategy that exits on HMM bear signals will naturally participate less in sustained one-directional trends. Negative alpha in a strong bull year is not indicative of a bug.

**Max drawdown context:** -7.56% max drawdown with 19.49% return gives a Calmar ratio of 2.58, which is favorable. Given 87 trades, a win rate near 50%, and avg win/loss ratio of 1.60 (consistent with a positive expectancy system), the drawdown figure is plausible and not suspicious.

**NaN trade duration:** The NaN in `avg_trade_duration_bars` is a vectorbt column lookup issue. It does not affect the validity of any other metric.

---

## Recommendation

**Status: CONDITIONAL PASS**

The strategy is free of critical lookahead bias and data leakage. It may proceed to the Analytics Reviewer phase with the following caveats noted for remediation:

1. **Fix W2 (Spread not applied):** Incorporate the 3-pip spread into the slippage or fees calculation so that reported costs match the declared `SPREAD_PIPS = 3`. Current performance is optimistic by approximately $0.03 per unit per trade.

2. **Fix W4 (Walk-forward re-training):** If walk-forward results are used as a decision criterion, re-fit the HMM on each window's training portion to produce a genuinely out-of-sample walk-forward test.

3. **Investigate W5 (NaN duration):** Resolve the vectorbt column name mismatch so that trade duration is captured and can inform risk management review.

4. **Document W1 (FVG fragility):** Add explicit comments to the FVG section noting that the `.shift(1)` on the final output is the sole lookahead guard, and that removing it would introduce bias.

5. **Accept W3 (Static slippage) as low-risk:** The error from using a period-average price for slippage is within 15% of the slippage cost and is acceptable for an initial audit pass.

The test-period APPROVED verdict (all 6 pass criteria met) is credible given the absence of bias issues and the realistic metric profile. No metric triggers the suspicion flags defined in the audit checklist.

---

*Report generated by Strategy Debugger Agent | 2026-03-09*
