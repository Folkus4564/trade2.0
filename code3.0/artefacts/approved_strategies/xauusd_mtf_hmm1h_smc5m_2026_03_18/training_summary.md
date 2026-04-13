# Strategy Export: xauusd_mtf_hmm1h_smc5m

**Export date**: 2026-03-18
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 30.09% | 31.94% | 89.33% |
| Sharpe Ratio | 2.475 | 2.818 | 4.100 |
| Max Drawdown | -6.21% | -3.58% | -7.53% |
| Profit Factor | 2.121 | 2.348 | 3.290 |
| Total Trades | 518 | 94 | 128 |
| Win Rate | 48.07% | 55.32% | 57.81% |

## Parameters

```json
{
  "hma_period": 55,
  "ema_period": 21,
  "atr_period": 14,
  "rsi_period": 14,
  "adx_period": 14,
  "dc_period": 40,
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

- Mean Sharpe: 1.430
- Positive Windows: 100.00%
- Windows: 7