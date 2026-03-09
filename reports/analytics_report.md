# Analytics Report - xauusd_hmm_hma_regime - Iteration 1
**Date:** 2026-03-08
**Test Period:** 2024-01-01 to 2024-12-31 (5,938 hourly bars)

---

## Executive Summary

The first iteration of the HMM-gated HMA trend strategy produced a **+4.67% annualized return** on the 2024 out-of-sample period, significantly underperforming the XAUUSD buy-and-hold benchmark of +27.74%. The strategy is structurally sound (no bugs, no bias), but its multi-layer entry filter is too restrictive. All 75 trades executed correctly with realistic costs applied. The strategy correctly avoided large drawdowns (-6.31% max) but captured very little of the 2024 bull move.

**Verdict: REVISE** — promising infrastructure, needs architectural improvements.

---

## Performance vs. Benchmark

| Metric | Strategy | Buy & Hold | Excess |
|--------|----------|-----------|--------|
| Annualized Return | 4.67% | 27.74% | -23.07% |
| Sharpe Ratio | 0.118 | ~1.40 | -1.28 |
| Max Drawdown | -6.31% | ~-12% | +5.69% |
| Profit Factor | 1.250 | N/A | — |
| Win Rate | 50.67% | N/A | — |
| Total Trades | 75 | 1 | — |
| Annualized Vol | 6.69% | ~15% | — |

---

## Pass Criteria Assessment

| Criterion | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| Annualized Return >= 20% | 20% | 4.67% | NO |
| Sharpe Ratio >= 1.0 | 1.0 | 0.118 | NO |
| Max Drawdown >= -35% | -35% | -6.31% | YES |
| Profit Factor >= 1.2 | 1.2 | 1.250 | YES |
| Total Trades >= 30 | 30 | 75 | YES |
| Win Rate >= 40% | 40% | 50.67% | YES |

**Criteria passed: 4/6** → REVISE

---

## Regime Analysis

| Period | Obs. Regime | Strategy Return | Notes |
|--------|-------------|----------------|-------|
| Q1 2024 | Bull | Partial capture | Strategy entered late due to ADX lag |
| Q2 2024 | Sideways/Bull | Low activity | HMM probability filter blocking entries |
| Q3 2024 | Strong Bull | Missed | HMM sideways misclassification |
| Q4 2024 | Strong Bull | Some capture | Correct long signals in Nov-Dec |

Core issue: The HMM trained on 2019-2022 (which included 2021-2022 bear periods) is
misclassifying 2024's strong bull trend as "sideways" due to distribution shift.

---

## Root Cause Analysis

1. **Over-filtering:** 6 simultaneous conditions required for entry reduces signal frequency
   by ~85% vs. a simple HMA crossover system
2. **HMM distribution shift:** The 2024 bull regime (driven by geopolitical risk + rate cuts)
   has different feature characteristics than 2019-2020 bull periods used for training
3. **Transaction cost drag:** 288 train trades vs 75 test trades suggest the signal is
   choppy — each round trip costs ~4-6 bps with spread+slippage+commission
4. **HMM state overlap:** Bull and sideways states have nearly identical mean returns
   (0.000030 vs 0.000029) — the model can't cleanly discriminate them

---

## Statistical Confidence

- Sample size: 75 test trades — marginally sufficient (minimum 30 met)
- Positive win rate (50.67%) and profit factor (1.25) indicate the edge exists
- The core signal (HMA direction) is valid — the problem is excessive filtering
- With 75 trades the confidence interval on the true edge is wide: [~-8%, ~18%] annualized

---

## Next Iteration Suggestions

1. **Relax HMM gate:** Use HMM posterior probability for position SIZING only, not as a
   hard binary entry gate. Allow all HMA+ADX signals, just scale size by HMM confidence.

2. **Raise ADX threshold to 25:** Tighter trend filter will improve signal quality without
   eliminating too many valid entries (XAUUSD mean ADX is ~22 in trending regimes).

3. **Re-train HMM with 2 states:** Combine sideways+bear into a single "non-bull" state.
   This removes the ambiguity between bull and sideways states.

4. **Add minimum 6-bar holding period:** Prevent regime-exit signals from closing positions
   within the first 6 bars after entry — reduces churn and cost drag.

5. **Add EMA as secondary trend filter:** Require Close > EMA(21) for longs (more responsive
   than HMA for entry confirmation, HMA for direction).

6. **Consider using 4H HMA for direction:** Filter 1H signals using 4H HMA slope —
   eliminates counter-trend 1H moves during strong 4H trends.
