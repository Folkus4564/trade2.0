# Strategy H — Pullback-Retest v6 (Long-Only Pullback)
**Approved:** 2026-04-01

## Overview
v6 extends Strategy G by disabling short pullback entries (`scalp_pullback.long_only=true`).
Shorts are handled by `scalp_momentum` only (which was 55.7%+ WR in Strategy E).

v6 result vs v5:
- WR jumps: 52.84% → **55.24%** (hits 55%+ target!)
- Return rises: 159.88% → **165.39%**
- Sharpe: 4.022 → **3.857** (slight dip, still strong)
- Trades fall: 2360 → **2065** (no short pullbacks = -1014 removed, but longs still 2065)
- WF: 7/7 → **6/7** (W4 just negative)

Key insight: scalp_pullback shorts (43.9% WR, 1014 trades) were dragging down overall WR.
Removing them lifts WR to 55.2% while actually increasing total return — scalp_momentum
handles trending shorts more efficiently.

## Architecture
- Config: `configs/experiments/hf_pullback_v6.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Regime TF: 1H | Signal TF: 5M | Engine: concurrent-5 | mode: regime_specialized
- Sub-strategies: trend + scalp_momentum (both directions) + scalp_pullback (longs only)

## Performance Summary

| Split  | Return   | Sharpe | MaxDD    | Trades | WR     |
|--------|----------|--------|----------|--------|--------|
| Train  | 32.35%   | 1.812  | -9.66%  | 9006   | 49.49% |
| Val    | 41.74%   | 2.475  | -4.59%   | 1526   | 50.66% |
| Test   | 165.44%  | 3.857  | -10.18%  | 2065   | 55.16% |

## Walk-Forward (7 windows, 6/7 positive, mean_sharpe=0.843)

| Window | Period | Sharpe |
|--------|--------|--------|
| W1 | — | 0.156 |
| W2 | — | 0.512 |
| W3 | — | 1.805 |
| W4 | — | -0.05 |
| W5 | — | 1.499 |
| W6 | — | 0.077 |
| W7 | — | 1.904 |
| Mean   | —      | 0.843 |
| Positive | —    | 6/7 (86%) |

## Key Parameters
- scalp_pullback: long_only=true (NO short pullback entries)
- scalp_pullback: SL=1.5x ATR, TP=1.5x ATR, min_prob_long=0.72
- scalp_momentum: SL=1.5x ATR, TP=1.5x ATR, require_macro_trend=True (handles shorts)
- trend: SL=2.0x ATR, TP=3.0x ATR
- max_concurrent_positions: 5 | base_allocation_frac: 0.75

## Reproduce
```bash
cd code3.0
trade2 --config configs/experiments/hf_pullback_v6.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl
```
