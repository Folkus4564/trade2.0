# Strategy Export: xauusd_5m_sd_ob_pullback_bidir_v1

**Export date**: 2026-04-06
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 12.53% | 39.09% | 165.03% |
| Sharpe Ratio | 0.422 | 1.235 | 2.339 |
| Max Drawdown | -31.03% | -16.70% | -18.60% |
| Profit Factor | 1.047 | 1.111 | 1.232 |
| Total Trades | 9155 | 1655 | 2165 |
| Win Rate | 54.92% | 55.77% | 58.29% |

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
  "hmm_min_prob": 0.55,
  "hmm_states": 3,
  "regime_persistence_bars": 3,
  "atr_stop_mult": 1.5,
  "atr_tp_mult": 1.5,
  "require_smc_confluence": true,
  "require_pin_bar": false
}
```