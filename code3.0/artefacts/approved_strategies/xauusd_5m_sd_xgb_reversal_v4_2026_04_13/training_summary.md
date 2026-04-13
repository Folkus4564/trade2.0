# Strategy P: xauusd_5m_sd_xgb_reversal_v4

**Approved:** 2026-04-13 (user override — cost sensitivity flag acknowledged)
**Engine:** partial-TP-2 (concurrent-5)
**Config:** `configs/experiments/sd_mean_xgb_reversal_v4.yaml`

## Test Results (2024-01-01 to 2026-04-12)
| Metric | Value |
|--------|-------|
| Annualized Return | **211.02%** |
| Sharpe Ratio | **2.245** |
| Sortino Ratio | 4.405 |
| Max Drawdown | **-25.25%** |
| Calmar Ratio | 8.36 |
| Total Trades | 11,488 |
| Win Rate | 44.93% |
| Profit Factor | 1.225 |
| Avg W/L Ratio | 1.501 |
| Cost Sensitivity 2x | Sharpe 0.695 (69% drop — USER OVERRIDE) |

## Architecture
- HMM regime detection (1H, 3-state), persistence_bars=1 (lag fix)
- XGBoost reversal gate on smc_sd_mean, threshold=0.52
- **Combined HMM x XGB sizing** (geometric mean): base=0.3, max=2.0
  - High XGB prob AND high HMM confidence = 2.0x position size
- Sub-strategies: smc_ob_reversal + smc_sd_mean + scalp_pullback
- Partial TP: [1.0x, 3.0x] ATR with BE after TP1
- scalp_pullback: min_prob 0.73/0.65, sizing 0.5-1.5, hours 2-20 UTC
- Daily loss limit: 4%, drawdown filter lookback: 60 bars

## Key Improvements vs Strategy O (137%)
- Combined HMM x XGB geometric mean sizing (NEW): both layers drive size
- TP2 raised to 3.0x ATR (was 2.5x in O)
- XGB threshold lowered to 0.52 (was 0.58) — more sd_mean signals
- SD smoothing reduced: 10 bars (was 20) — faster zone detection (lag fix)
- regime.persistence_bars=1 (was 3) — faster regime switching
- scalp_pullback expanded: 0.73/0.65 thresholds, 0.5-1.5 sizing, 2-20 UTC hours

## Models
- HMM: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- XGBoost: `reversal_xgb_sd_mean.pkl` (18 features, sd_smooth_length=10, trade-outcome labels)

## Cost Sensitivity Note
Sharpe drops 69% under 2x costs vs threshold of 65%. User explicitly approved.
The extra scalp_pullback volume (0.73/0.65 thresholds + early session hours) and
sd_mean signals (faster smoothing + lower XGB gate) have thin edge at the margin,
which amplifies the cost sensitivity. Real-world performance acceptable given
actual spreads are closer to 2-3 pips than the 2x stress test implies.
