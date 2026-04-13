# Strategy Export: xauusd_5m_pullback_v6_smc_ob_v1

**Export date**: 2026-04-05
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 32.42% | 41.71% | 165.89% |
| Sharpe Ratio | 1.815 | 2.473 | 3.864 |
| Max Drawdown | -9.58% | -4.59% | -10.18% |
| Profit Factor | 1.294 | 1.399 | 1.729 |
| Total Trades | 9034 | 1530 | 2070 |
| Win Rate | 49.51% | 50.59% | 55.12% |

## Parameters

```json
{
  "hma_period": 55,
  "ema_period": 21,
  "atr_period": 14,
  "rsi_period": 14,
  "adx_period": 14,
  "dc_period": 8,
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