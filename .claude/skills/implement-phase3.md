# Phase 3: Probabilistic HMM — Use Posterior Regime Probabilities for Gating and Risk Scaling

## Context
Part of the 6-phase roadmap from `memory/improvement_suggestions_checkpoint.md`.
Phases 1 and 2 must be complete before starting this phase.

## Goal
Replace the hard regime label gate (`hmm_label == "bull"`) with a soft, probability-weighted
approach. Entry permission and position size both scale continuously with regime confidence,
eliminating the binary on/off cliff edge that currently causes signal clustering near the threshold.

## Problem Statement
Current code:
1. `hard_rejection.py` treats HMM as a hard gate: label must be "bull" or "bear"
2. `min_prob_hard` threshold is a cliff — a bar at 0.769 fires, 0.768 does not
3. Position sizing (`hmm/sizing_base` + `hmm/sizing_max`) is currently a linear interpolation
   but uses `bull_probability` from `hmm.bull_probability()` — check if this is actually wired up
4. No penalty for uncertainty: bars where no regime has probability > 0.5 are not filtered out

## Changes

### 1. Verify existing probability wiring in signals/generator.py
**File**: `code3.0/src/trade2/signals/generator.py`

Read the file. Check:
- Is `hmm_bull_prob` used for position sizing or just the label?
- Is there a `position_size_long` / `position_size_short` column generated from probabilities?
- If sizing is flat (not probability-weighted), add probability-weighted sizing.

### 2. Add soft regime gate: max_entropy filtering
**File**: `code3.0/src/trade2/signals/generator.py`

Add a new filter: block entries when the HMM state is uncertain.
- Uncertain = max(bull_prob, bear_prob, sideways_prob) < `hmm.min_confidence` threshold
- Add `hmm.min_confidence: 0.55` to `configs/base.yaml` under `hmm` section
- If max probability across all states < min_confidence, suppress signal (set to 0)

This catches transition bars where the HMM is split between states even if the leading state
passes `min_prob_hard`.

### 3. Probability-weighted position sizing
**File**: `code3.0/src/trade2/signals/generator.py`

Current sizing: `position_size = sizing_base + (prob - min_prob_hard) * sizing_scale` (if implemented).
Target sizing formula:
```
prob_excess = max(0, regime_prob - min_confidence)
prob_range  = 1.0 - min_confidence
size_mult   = sizing_base + (prob_excess / prob_range) * (sizing_max - sizing_base)
size_mult   = clip(size_mult, sizing_base, sizing_max)
```
This gives `sizing_base` at the minimum confidence threshold, `sizing_max` at full certainty (prob=1.0).

### 4. Add min_confidence to base.yaml
**File**: `code3.0/configs/base.yaml`

Under `hmm` section, add:
```yaml
  min_confidence: 0.55    # minimum max-state-probability to allow any entry
```

### 5. Uncertainty cooldown (transition penalty)
**File**: `code3.0/src/trade2/signals/generator.py`

After a regime flip (label changes from previous bar), suppress entries for `regime.transition_cooldown_bars` bars.
Add to `configs/base.yaml`:
```yaml
regime:
  transition_cooldown_bars: 2   # bars to skip after regime flip
```

### 6. Log regime probability stats in pipeline
**File**: `code3.0/src/trade2/app/run_pipeline.py`

After generating signals, print:
- Mean bull_prob / bear_prob on signal bars vs non-signal bars
- Fraction of bars filtered by min_confidence
- Fraction of bars filtered by transition_cooldown

## Files to Modify
1. `code3.0/src/trade2/signals/generator.py` — soft gate, prob-weighted sizing, transition cooldown
2. `code3.0/configs/base.yaml` — add hmm.min_confidence, regime.transition_cooldown_bars
3. `code3.0/src/trade2/app/run_pipeline.py` — log regime probability stats

## Verification
1. `trade2 --retrain-model --skip-walk-forward`
2. Verify signal count is different (uncertainty filter will reduce some signals)
3. Check that position sizes are NOT all the same — print min/max/mean of position_size_long on signal bars
4. Check that no signals exist within transition_cooldown_bars after a regime flip
5. Compare test Sharpe vs Phase 2 baseline — expect more stable, possibly lower trade count

## Before Starting
- Read all files before making changes
- Check memory/MEMORY.md for current results baseline (Phase 2 results)
- Check `code3.0/src/trade2/models/hmm.py` to understand bull_probability() and bear_probability() return shapes
