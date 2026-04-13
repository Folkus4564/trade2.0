# Strategy Export: xauusd_mtf_hmm1h_smc5m

**Export date**: 2026-03-17
**Verdict**: HARD_REJECTED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 10.39% | 53.73% | 25.96% |
| Sharpe Ratio | 0.483 | 2.877 | 2.392 |
| Max Drawdown | -25.83% | -5.73% | -5.68% |
| Profit Factor | 1.308 | 4.531 | 4.829 |
| Total Trades | 140 | 24 | 18 |
| Win Rate | 55.71% | 79.17% | 83.33% |

## Parameters

```json
{
  "hma_period": 55,
  "ema_period": 21,
  "atr_period": 14,
  "rsi_period": 14,
  "adx_period": 14,
  "dc_period": 40,
  "adx_threshold": 16.715103754797116,
  "hmm_min_prob": 0.7348404387333658,
  "hmm_states": 2,
  "regime_persistence_bars": 4,
  "atr_stop_mult": 3.095514338084446,
  "atr_tp_mult": 4.488045570804698,
  "require_smc_confluence": true,
  "require_pin_bar": false
}
```