# Phase 2: Strategy Simplification — Reduce Noisy Features, Define Clear Regime Behavior

## Context
Part of the 6-phase roadmap from `memory/improvement_suggestions_checkpoint.md`.
Phase 1 (cost modelling + config-driven evaluation) is complete.

## Goal
Reduce feature noise, remove weak SMC signals, and clearly separate what the strategy does
in each of the 3 HMM regimes (bull/trend, bear/range, sideways/no-trade).

## Problem Statement
Current code has:
1. **Too many SMC features** — OB, FVG, sweep_low, sweep_high, pin_bar; several rarely fire and add noise
2. **No regime-specific entry logic** — same entry conditions used regardless of whether HMM says trend, range, or sideways
3. **signals/generator.py is monolithic** — trend + range logic tangled together; hard to tune or reason about
4. **Noisy feature list** — `hmm_features.py` uses 7 features including `hmm_feat_bb_width` and `hmm_feat_macd` which are correlated with other features

## Changes

### 1. Audit and prune SMC features
**File**: `code3.0/src/trade2/features/smc.py`

Read the file first. Check which of these features have < 2% hit rate in train data:
- `ob_bullish`, `ob_bearish` — order blocks
- `fvg_bullish`, `fvg_bearish` — fair value gaps
- `sweep_low`, `sweep_high` — liquidity sweeps

Prune any feature with < 2% hit rate on train split or with < 0.5 correlation to forward 1-bar return.
Log which features were kept/removed.

### 2. Simplify HMM feature set
**File**: `code3.0/src/trade2/features/hmm_features.py`

Current 7-feature set: `hmm_feat_ret, hmm_feat_rsi, hmm_feat_atr, hmm_feat_vol, hmm_feat_hma_slope, hmm_feat_bb_width, hmm_feat_macd`

- Remove `hmm_feat_bb_width` (correlated with `hmm_feat_atr`)
- Remove `hmm_feat_macd` (correlated with `hmm_feat_ret` + `hmm_feat_hma_slope`)
- Keep: `hmm_feat_ret, hmm_feat_rsi, hmm_feat_atr, hmm_feat_vol, hmm_feat_hma_slope`
- Update `configs/base.yaml` `hmm.features` list to match

### 3. Regime-specific signal gating in generator.py
**File**: `code3.0/src/trade2/signals/generator.py`

Read the file first. Add regime-aware gating:
- **bull regime**: allow long entries only (no shorts); require `hma_rising=True` AND `price_above_hma=True`
- **bear regime**: allow short entries only (no longs); require `hma_rising=False` AND `price_above_hma=False`
- **sideways regime**: no entries (skip signal generation entirely)

This replaces the current approach of allowing both directions in all regimes filtered only by HMM probability threshold.

Add a new config key `regime.allow_counter_trend: false` in `base.yaml` (default false = enforce regime-direction alignment).

### 4. Add regime label to trade records
**File**: `code3.0/src/trade2/backtesting/engine.py`

At entry time, record the active HMM regime label in the trades dict so post-analysis can show performance by regime.
Requires passing `regime_labels` array (same length as df) into `_simulate_trades()`.
Add `regime_labels: np.ndarray = None` as optional param. If provided, store `regime_labels[entry_bar]` in each trade dict as `"regime"`.

### 5. Update base.yaml
Add to `regime` section:
```yaml
regime:
  allow_counter_trend: false
```

## Files to Modify
1. `code3.0/src/trade2/features/smc.py` — audit + prune low-signal features
2. `code3.0/src/trade2/features/hmm_features.py` — drop bb_width and macd
3. `code3.0/configs/base.yaml` — update hmm.features list + add regime.allow_counter_trend
4. `code3.0/src/trade2/signals/generator.py` — regime-specific entry gating
5. `code3.0/src/trade2/backtesting/engine.py` — optional regime label in trade records

## Verification
1. `trade2 --retrain-model --skip-walk-forward` — runs without errors
2. Check train signal counts: expect similar or slightly fewer signals (pruned noise)
3. Verify no longs generated in bear regime and no shorts in bull regime
4. Compare test metrics vs Phase 1 baseline (expect stable or improved Sharpe due to cleaner signals)
5. Check `hmm_features` list in retrained model has 5 features not 7

## Before Starting
- Read all files listed above before making any changes
- Check current SMC feature hit rates by running a quick analysis on train_1h or train_5m
- Check memory/MEMORY.md and memory/improvement_suggestions_checkpoint.md for context
- After completing, update memory/MEMORY.md with new results
