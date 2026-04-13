# Strategy: xauusd_hf_concurrent3_105pct
**Approved:** 2026-03-30
**Status:** Approved
**Folder:** `artefacts/approved_strategies/xauusd_hf_concurrent3_105pct_2026_03_30/`

---

## Performance Summary

| Split | Return | Sharpe | Max DD | Trades | Win Rate |
|-------|--------|--------|--------|--------|----------|
| Train (2019-2022) | 28.67% | 1.837 | -7.79% | 4788 | 45.86% |
| Val (2023) | 30.29% | 2.183 | -4.80% | 809 | 48.21% |
| **Test (2024-2025)** | **105.44%** | **3.930** | **-9.98%** | **1069** | **51.73%** |
| Test @ 2x costs | 86.67% | 3.389 | -10.26% | 1069 | — |

**Trades per day (test):** 3.45
**Walk-Forward:** 7/7 windows positive (100%) | mean_sharpe = 0.688

---

## Architecture

**Mode:** `regime_specialized` — two active sub-strategies routing on HMM regime probability.

**Sub-strategies:**
- `trend`: Donchian Channel breakout in HMM bull/bear regime. SL=2.0x ATR, TP=3.0x ATR.
- `scalp_momentum`: 8-bar DC breakout on 5M bars. SL=1.0x ATR, TP=1.5x ATR. Provides the majority of trade frequency.

**HMM:** 36-feature GaussianHMM, 3 states, trained on 1H XAUUSD 2019-2022.
**Model:** `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**ATR expansion gate:** Both strategies require ATR above its 20-bar rolling mean to enter (ratio=1.0). This quality gate eliminates low-volatility false breakouts.

---

## Key Innovation: Multi-Position Concurrent Engine

This strategy uses the **concurrent backtest engine** (`_simulate_trades_multi` in `src/trade2/backtesting/engine.py`).

**Problem solved:** The original single-position engine was blocking ~74% of quality signals because a position was already open. With ~5 quality signals/day firing in clusters, only ~1.29 trades/day were actually executed.

**Solution:** Set `risk.max_concurrent_positions: 3` in config. The engine allows up to 3 simultaneous positions. Capital exposure is held constant: each position uses `base_allocation_frac / 3` of capital.

**Effect:**
- Trades/day: 1.29 → 3.45
- Signal utilization: ~26% → ~69%
- WF window 6 (2024 H1 parabolic rally): negative → +1.39% (previously unfixable)
- WF result: 86% positive → 100% positive (7/7 windows)

**Risk is NOT increased:** Total capital-at-risk = same as single-position engine. The same `base_allocation_frac` budget is just spread across up to 3 concurrent smaller positions instead of 1 large one.

---

## How to Reproduce

```bash
cd code3.0

# Full run with walk-forward (recommended)
trade2 --config configs/hf_concurrent3_105pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

# Quick check (no WF)
trade2 --config configs/hf_concurrent3_105pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl \
       --skip-walk-forward
```

Expected output:
```
TRAIN  | Return: 28.67% | Sharpe: 1.837 | MaxDD: -7.79%  | Trades: 4788
VAL    | Return: 30.29% | Sharpe: 2.183 | MaxDD: -4.80%  | Trades: 809
TEST   | Return: 105.44% | Sharpe: 3.930 | MaxDD: -9.98% | Trades: 1069
WF     | 7/7 positive (100%) | mean_sharpe: 0.688
VERDICT: APPROVED
```

---

## Config Key Parameters

```yaml
risk:
  max_concurrent_positions: 3      # CRITICAL: enables concurrent engine
  base_allocation_frac: 0.95
  max_hold_bars: 0

strategies:
  mode: regime_specialized
  scalp_momentum:
    enabled: true
    dc_period: 8
    atr_sl_mult: 1.0
    atr_tp_mult: 1.5
    atr_expansion_filter: true
    atr_expansion_ratio: 1.0
  trend:
    enabled: true
    dc_period: 10
    atr_sl_mult: 2.0
    atr_tp_mult: 3.0
    atr_expansion_filter: true
```

---

## Relationship to 122% Strategy (Strategy C)

This is a direct derivative of `xauusd_hf_r1p0_lb20_2026_03_29` (122% / Strategy C).
**Only one parameter differs:** `risk.max_concurrent_positions: 3` (was 1).

| Metric | 122% (single-pos) | 105% (concurrent-3) |
|--------|-------------------|---------------------|
| Test Return | 122.86% | 105.44% |
| Test Sharpe | 3.512 | 3.930 |
| Max DD | -12.17% | -9.98% |
| TPD | 1.29 | 3.45 |
| WF Positive | 86% (6/7) | **100% (7/7)** |
| WF mean_sharpe | 1.010 | 0.688 |

Trade-off: lower raw return but **better Sharpe, lower drawdown, higher trade frequency, and perfect WF score**.

---

## Walk-Forward Windows

| Window | Period | Return | Sharpe |
|--------|--------|--------|--------|
| 1 | 2021-01-01 to 2021-06-30 | 8.13% | 0.935 |
| 2 | 2022-01-01 to 2022-06-30 | 10.39% | 0.644 |
| 3 | 2022-07-01 to 2022-12-31 | 17.88% | 1.483 |
| 4 | 2023-01-01 to 2023-06-30 | 1.39% | 0.067 |
| 5 | 2023-07-01 to 2023-12-31 | 15.34% | 1.418 |
| 6 | 2024-01-01 to 2024-06-30 | 1.39% | +ve* |
| 7 | 2024-07-01 to 2024-12-31 | 5.86% | 0.212 |

*Window 6 was consistently negative for all single-position configs (gold parabolic rally, 34-38% WR on DC breakout). The concurrent engine turns it slightly positive by capturing trend signals even while scalp positions are still open.
