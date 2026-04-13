# Strategy Export: xauusd_5m_sd_ob_pullback_bidir_fixedtp

**Export date**: 2026-04-07
**Verdict**: APPROVED

## Performance Summary

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Annualized Return | 44.46% | 56.56% | 102.92% |
| Sharpe Ratio | 1.203 | 1.634 | 2.631 |
| Max Drawdown | -28.84% | -14.28% | -15.59% |
| Profit Factor | 1.133 | 1.160 | 1.262 |
| Total Trades | 15244 | 3112 | 4724 |
| Win Rate | 39.52% | 46.14% | 60.67% |

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