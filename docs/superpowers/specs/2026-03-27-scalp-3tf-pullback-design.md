# Scalp 3TF Pullback Strategy — Design Spec
**Date**: 2026-03-27
**Status**: Approved

---

## Problem

The existing 1M scalping strategy loses money:
- Win rate: 37.8%, Profit Factor: 0.342, Total PnL: -$28K over 14 months
- 62% of trades hit SL, 16.8% exit instantly (stop-hunted on 1M noise)
- Root cause: entering at signal bar close on 1M is chasing into noise

The 5M config is also losing (WR 35.5%, PF 0.665).

---

## Goal

A profitable scalping strategy with:
- 2+ trades/day
- Win rate > 45%
- Profit Factor > 1.2
- Sharpe > 1.5
- Max drawdown < -20%

---

## Architecture

Three-layer pipeline:

```
1H bars  →  HMM (3-state)       →  Regime label (Bull/Bear/Neutral)
                                            ↓ forward-fill to 5M and 1M
5M bars  →  Signal generator    →  Entry signal (Long/Short) + ATR_5m
                                            ↓ pass signal + ATR_5m to 1M layer
1M bars  →  PullbackEngine      →  Tiered limit order → fill → SL/TP/trail
```

**Key principle**: Regime quality from 1H (proven in 89% strategy). Signal direction from 5M (less noisy than 1M). Precise entry timing from 1M pullbacks. Stop sizing always from 5M ATR to avoid 1M noise.

**Data files:**
- Regime: `data/raw/XAUUSD_1H_2019_2026_full.csv`
- Signal: `data/raw/XAUUSD_5M_2019_2026.csv`
- Entry: `code3.0/data/raw/XAUUSD_1M_2019_2026.csv`

---

## Entry Logic — PullbackEngine

### Adaptive Level Definition
Tier multipliers are **regime-aware and Optuna-optimized** — not hardcoded.
Computed at 5M signal bar close:

```
# Step 1: pick per-regime base multipliers (from optimized config)
if regime == Bull:
    base_t1, base_t2 = bull_tier1_mult, bull_tier2_mult
elif regime == Bear:
    base_t1, base_t2 = bear_tier1_mult, bear_tier2_mult
else:  # Neutral/ranging
    base_t1, base_t2 = neutral_tier1_mult, neutral_tier2_mult

# Step 2: scale by volume ratio (high volume → deeper expected pullback)
vol_ratio = 5M_bar_volume / rolling_avg_volume(20 bars)
vol_factor = clamp(vol_ratio, 0.5, 2.0)

# Step 3: compute limits
effective_t1 = base_t1 × ATR_5m × vol_factor
effective_t2 = base_t2 × ATR_5m × vol_factor

Long signal:
  tier1_limit = signal_close - effective_t1
  tier2_limit = signal_close - effective_t2

Short signal:
  tier1_limit = signal_close + effective_t1
  tier2_limit = signal_close + effective_t2
```

**Constraint enforced**: `tier2_mult < tier1_mult` always (tier2 is shallower = easier fill).

### Per-1M-Bar Evaluation (strict order)
1. **Check invalidation** → if triggered, cancel order, stop processing
2. **Check tier1** → fill if `1M_low <= tier1_limit` (long) or `1M_high >= tier1_limit` (short)
3. **Check tier2** → fill if `1M_low <= tier2_limit` (long) or `1M_high >= tier2_limit` (short)
4. **Cancel** if bar count > `max_wait_bars` with no fill

Both tiers active from bar 1. Tier1 always checked first (better price). Whichever is hit first wins. Fill price is always the limit price (tier1_limit or tier2_limit), not the 1M bar low/high.

### Invalidation Conditions (cancel immediately)
- 1H regime flips away from signal direction (checked at each 1H close)
- Runaway: `1M_close > signal_close + runaway_atr_mult × ATR_5m` for longs (price left without us)
- Runaway: `1M_close < signal_close - runaway_atr_mult × ATR_5m` for shorts
- New 5M signal in **opposite** direction

`runaway_atr_mult` is Optuna-optimized in range [0.5, 2.0].

### Same-Direction Signal Update
New 5M signal in same direction while waiting → refresh tier1/tier2 levels and reset bar counter to 0. Do not open a second position.

---

## Risk & Exit Logic

### Stop and TP
Sized from **fill price** using **5M ATR** (not 1M ATR):

```
Long:
  SL = fill_price - 1.5 × ATR_5m
  TP = fill_price + 2.5 × ATR_5m   →  1:1.67 R:R

Short: flip signs
```

### Trailing Stop
- Activates after: price moves 0.8 × ATR_5m in favour
- Trails at: 0.8 × ATR_5m behind high-water mark
- Overrides TP if trailing stop locks in more profit

### Exit Priority (per 1M bar)
1. SL breach → exit at SL price
2. TP breach → exit at TP price
3. Trailing stop breach → exit at trail level
4. 1H regime flip → exit at next 1M open (signal exit)
5. Hold

No timeout (`max_hold_bars: 0`).

---

## New Files

| File | Purpose |
|------|---------|
| `code3.0/src/trade2/backtesting/pullback_engine.py` | PullbackEngine class |
| `code3.0/configs/scalp_3tf.yaml` | Config overlay for this strategy |

## Modified Files

| File | Change |
|------|--------|
| `code3.0/src/trade2/backtesting/engine.py` | Add 3-TF mode, delegate entry to PullbackEngine |
| `code3.0/src/trade2/signals/regime.py` | Forward-fill 1H regime to both 5M and 1M bars |
| `code3.0/src/trade2/app/run_pipeline.py` | Wire 3-TF data loading when `entry_timeframe` is set |

---

## Config (`scalp_3tf.yaml`)

```yaml
strategy:
  name: xauusd_scalp_3tf
  mode: multi_tf
  regime_timeframe: 1H
  signal_timeframe: 5M
  entry_timeframe: 1M

data:
  raw_1h_csv:  data/raw/XAUUSD_1H_2019_2026_full.csv
  raw_5m_csv:  data/raw/XAUUSD_5M_2019_2026.csv
  raw_1m_csv:  code3.0/data/raw/XAUUSD_1M_2019_2026.csv

pullback:
  # Per-regime multipliers — all Optuna-optimized
  bull_tier1_mult: 0.25
  bull_tier2_mult: 0.10
  bear_tier1_mult: 0.25
  bear_tier2_mult: 0.10
  neutral_tier1_mult: 0.15
  neutral_tier2_mult: 0.07
  # Volume scaling: effective_mult = base_mult × clamp(vol/avg_vol_20, 0.5, 2.0)
  max_wait_bars: 6
  runaway_atr_mult: 1.0

risk:
  atr_stop_mult: 1.5
  atr_tp_mult: 2.5
  use_signal_atr: true
  trailing_enabled: true
  trailing_atr_mult: 0.8
  break_even_atr_trigger: 0.8
  max_hold_bars: 0
  base_allocation_frac: 0.50

hmm:
  min_prob_hard: 0.45
  min_prob_hard_short: 0.45
  min_prob_entry: 0.45
  min_prob_exit: 0.40
  min_confidence: 0.40

session:
  enabled: true
  allowed_hours_utc: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

---

## Optimization Search Space

```yaml
optimization:
  n_trials: 200
  target: val_sharpe
  search_space:
    bull_tier1_mult:    [0.10, 0.50]
    bull_tier2_mult:    [0.05, 0.20]
    bear_tier1_mult:    [0.10, 0.50]
    bear_tier2_mult:    [0.05, 0.20]
    neutral_tier1_mult: [0.05, 0.30]
    neutral_tier2_mult: [0.02, 0.15]
    max_wait_bars:      [3, 10]
    runaway_atr_mult:   [0.5, 2.0]
    atr_stop_mult:      [1.0, 3.0]
    atr_tp_mult:        [1.5, 4.0]
    hmm_min_prob:       [0.35, 0.65]
    trailing_atr_mult:  [0.5, 1.5]
```

Constraint: `tiern_tier2_mult < tiern_tier1_mult` enforced per-trial (Optuna suggests, engine validates).

---

## Verification Steps

1. Run backtest on train set — confirm 2+ trades/day, no lookahead bias
2. Check fill distribution: what % fill at tier1 vs tier2 vs cancel
3. Check instant exits (0-bar trades) — should be near zero with pullback entries
4. Check vol_factor distribution — confirm it varies meaningfully (not always clamped)
5. Run Optuna optimization on val set — confirm best params improve over defaults
6. Run on test set — confirm Sharpe > 1.5, return > 30%, DD > -20%
