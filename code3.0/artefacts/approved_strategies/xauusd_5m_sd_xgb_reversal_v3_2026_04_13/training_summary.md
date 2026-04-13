# Strategy O: xauusd_5m_sd_xgb_reversal_v3

**Approved:** 2026-04-13  
**Engine:** partial-TP-2 (concurrent-5)  
**Config:** `configs/experiments/sd_mean_xgb_reversal_v3.yaml`

## Test Results (2024-01-01 to 2026-04-12)
| Metric | Value |
|--------|-------|
| Annualized Return | **137.45%** |
| Sharpe Ratio | **2.477** |
| Sortino Ratio | 4.17 |
| Max Drawdown | **-17.40%** |
| Calmar Ratio | 7.90 |
| Total Trades | 10,290 |
| Win Rate | 47.86% |
| Profit Factor | 1.271 |
| Avg W/L Ratio | 1.385 |
| Avg Trade Duration | 63.6 bars (~5.3h) |
| Cost Sensitivity 2x | Sharpe 0.969 (60.9% drop, PASSES) |

## Architecture
- HMM regime detection (1H, 3-state: bull/bear/sideways)
- XGBoost reversal probability gate on smc_sd_mean entries
  - Labels: TP1 (1.0x ATR) hit before SL (1.5x ATR) within 20 bars
  - Positive rate: ~57% (well-calibrated)
  - Threshold: 0.58 for both long and short
- Sub-strategies: smc_ob_reversal + smc_sd_mean + scalp_pullback
- Partial TP: [1.0x, 2.5x] ATR with BE after TP1
- Daily loss limit: 3%, drawdown filter lookback: 60 bars

## Key Improvements vs Strategy N
- XGB gate removes ~40% of sd_mean entries (poor reversal probability)
- Fixed sizing bug: scale from min_prob threshold (not min_confidence=0.45)
- TP2 raised to 2.5x (vs 1.5x in N) -> better avg W/L ratio (1.385 vs 0.818)
- Tightened short thresholds: ob_reversal 0.55, sd_mean 0.40, scalp_pullback 0.68
- Cost sensitivity: 60.9% drop vs N's ~62% -> both pass

## Models
- HMM: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- XGBoost: `reversal_xgb_sd_mean.pkl` (18 features, trade-outcome labels)
