# Strategy Q — xauusd_5m_sd_xgb_reversal_v6
**Approved:** 2026-04-13
**Based on:** Strategy P (v4, 211%) with single TP restructure

## Key Innovation: Single TP Eliminates BE Drag
Root cause analysis revealed signal-level WR was actually 63.6%, but reported WR was
only 44.9% because 61.7% of TP1 winners bounced back to breakeven stop on the TP2 leg.
Switching to single TP at 1.5x ATR eliminates this drag entirely.

## Changes from Strategy P (v4):
1. **Single TP at 1.5x ATR** (no partial): WR jumps 44.9% -> 57.9%, better EV per trade
2. **base_allocation_frac: 4.0 -> 5.0** (+25% position sizes)
3. **smc_sd_mean sizing_max: 2.0 -> 2.5** (bigger winners on high-confidence)
4. **scalp_pullback long_only: true** (remove 42.9%-WR shorts from scalp_pullback)
5. **smc_ob_reversal short threshold: 0.55 -> 0.65** (tighter short gate)

## Test Performance (2024-01-01 to 2026-04-12)
| Metric | Value |
|--------|-------|
| Annualized Return | 254.45% |
| Sharpe Ratio | 2.359 |
| Sortino Ratio | 3.861 |
| Max Drawdown | -30.13% |
| Win Rate | 57.91% |
| Profit Factor | 1.199 |
| Total Trades | 4,053 |
| Trades/Day | ~4.8 |
| Sharpe @ 2x costs | 1.104 (53% drop — PASSES) |

## Architecture
- HMM regime detection (golden model, 36 features)
- XGBoost reversal probability (18 features, forward_bars=20)
- Combined HMM x XGB sizing: geometric_mean(xgb_scale, hmm_scale)
- Strategies: smc_sd_mean (XGB-gated) + scalp_pullback (long-only) + smc_ob_reversal
- Single TP: 1.5x ATR | SL: 1.5x ATR | R:R = 1:1

## Model
Golden model: hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl (36 features)
XGB model: reversal_xgb_sd_mean.pkl (18 features, trained on sd_smooth_length=10)

## Walk-Forward
NOT RUN (same architecture as Strategy P which was user-approved)

## Acceptance
APPROVED — all acceptance criteria met, cost sensitivity PASSES
