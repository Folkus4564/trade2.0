# Strategy Export: xauusd_hf_r1p0_lb20

**Export date**: 2026-03-29
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 35.89% | 45.34% | 122.86% |
| Sharpe Ratio | 1.992 | 2.784 | 3.512 |
| Max Drawdown | -10.60% | -6.96% | -12.17% |
| Profit Factor | 1.353 | 1.549 | 1.786 |
| Total Trades | 1981 | 322 | 406 |
| Win Rate | 43.06% | 45.34% | 49.51% |

## Parameters

```json
{
  "hma_period": 55,
  "ema_period": 21,
  "atr_period": 14,
  "rsi_period": 14,
  "adx_period": 14,
  "dc_period": 10,
  "adx_threshold": 15.0,
  "hmm_min_prob": 0.77,
  "hmm_states": 3,
  "regime_persistence_bars": 3,
  "atr_stop_mult": 2.75,
  "atr_tp_mult": 15.0,
  "require_smc_confluence": true,
  "require_pin_bar": false
}
```

## Walk-Forward Results

- Mean Sharpe: 1.010
- Positive Windows: 100.00%
- Windows: 7