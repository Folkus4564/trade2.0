# Strategy Export: xauusd_mtf_hmm1h_smc5m

**Export date**: 2026-03-18
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 22.25% | 21.42% | 49.27% |
| Sharpe Ratio | 2.128 | 2.171 | 3.164 |
| Max Drawdown | -4.85% | -3.32% | -7.15% |
| Profit Factor | 1.573 | 1.509 | 1.815 |
| Total Trades | 909 | 162 | 221 |
| Win Rate | 59.85% | 61.11% | 62.44% |

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
  "atr_tp_mult": 2.0,
  "require_smc_confluence": true,
  "require_pin_bar": false
}
```

## Walk-Forward Results

- Mean Sharpe: 0.527
- Positive Windows: 85.71%
- Windows: 7