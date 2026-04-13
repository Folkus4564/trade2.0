# trade2.0 - XAUUSD Automated Quant Research System

## Overview
Self-operating quantitative trading research pipeline for XAUUSD (Gold).
Uses HMM regime detection + SMC signals on 1H/5M data.

## Quick Start
```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0

trade2 --config configs/experiments/hf_pullback_v6_smc_ob.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

trade2 --retrain-model --skip-walk-forward   # retrain + skip WF
trade2-live                                   # run live strategies
```

## Data Splits
- Train:      2019-01-01 to 2022-12-31
- Validation: 2023-01-01 to 2023-12-31
- Test:       2024-01-01 onwards

## Research Standards
- NO lookahead bias: all features shift(1) before HMM input
- NO data leakage: scaler/HMM fitted on train only
- Costs: 3 pip spread + 1 pip slippage + 2 bps commission
- Minimum 30 trades for statistical validity
- Walk-forward validation mandatory for any optimization

## Performance Targets
| Metric | Minimum | Target |
|--------|---------|--------|
| Annualized Return | 20% | 50%+ |
| Sharpe Ratio | 1.0 | 1.5+ |
| Max Drawdown | -35% | -20% |
| Win Rate | — | 55%+ |
| Trades/Day | — | 5+ |

## User Preferences
- Auto-proceed without asking for confirmation
- No unicode special characters in print statements (cp874 encoding)
- Custom event-driven engine (pandas/numpy, no vectorbt)
- Always push to GitHub after committing
- NO timeout exits — `max_hold_bars: 0` in all configs

## Golden Model Protection
- `--retrain-model` auto-backs up old model to `artefacts/models/backups/`
- Any run with test return >= 20% auto-saves to `artefacts/models/golden/`
- `--model-path <pkl>` reuses any specific model without retraining
- Threshold: `pipeline.golden_model_threshold` in base.yaml (default 0.20)

## APPROVED STRATEGY PROTECTION — ABSOLUTE RULE
**NEVER modify any file inside `code3.0/artefacts/approved_strategies/`.**

| Strategy | Return | Sharpe | DD | WF | Engine | Status |
|----------|--------|--------|----|----|--------|--------|
| A (89%) | 89.33% | 4.10 | -7.53% | 100% | single | Live-deployed |
| B (49%) | 49.27% | 3.16 | -7.15% | 100% | single | Live-deployed |
| C (122%) | 122.86% | 3.51 | -12.17% | 100% | single | Approved 2026-03-29 |
| D (105%) | 105.44% | 3.93 | -9.98% | 100% | concurrent-3 | Approved 2026-03-30 |
| E (115%) | 115.30% | 4.18 | -9.65% | 86% | concurrent-5 | Approved 2026-03-30 |
| H (165%) | 165.44% | 3.86 | -10.18% | 86% | concurrent-5 | Approved 2026-04-01 |
| I (166%) | 165.89% | 3.86 | -10.18% | 100% | concurrent-5 | **NEW BEST** 2026-04-05 |
| J (54%) | 53.99% | 1.86 | -12.92% | N/A | concurrent-5 | Approved 2026-04-06 |
| M (165%) | 165.03% | 2.34 | -18.60% | 86% | concurrent-5 | Approved 2026-04-06 |
| N (103%) | 102.92% | 2.63 | -15.59% | N/A | partial-TP-2 | Approved 2026-04-07 |
| O (137%) | 137.45% | 2.477 | -17.40% | N/A | concurrent-5 | Approved 2026-04-13 |
| P (211%) | 211.02% | 2.245 | -25.25% | N/A | concurrent-5 | Approved 2026-04-13 (user override) |
| Q (254%) | 254.45% | 2.359 | -30.13% | N/A | concurrent-5 | **NEW BEST** 2026-04-13 |

Golden model (all strategies): `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Rules:**
- Do NOT edit `config.yaml`, `metrics.json`, `trades_test.csv`, or `training_summary.md` in any folder.
- Do NOT delete or overwrite `model.pkl` in any folder.
- Any improvement work must go to a NEW folder with a new timestamp name.

**Recommended config:** `configs/experiments/sd_mean_xgb_reversal_v6.yaml` — 254.5%, WR=57.9%, ~4.8 TPD, cost-sensitivity PASSES

## Production Configs (ranked)
| Config | Return | Sharpe | DD | TPD | WF |
|--------|--------|--------|----|-----|----|
| `experiments/hf_pullback_v6_smc_ob.yaml` | 165.89% | 3.864 | -10.18% | ~6.6 | 100% |
| `experiments/hf_pullback_v6.yaml` | 165.44% | 3.857 | -10.18% | ~6.6 | 86% |
| `hf_macro_sl15_55pct.yaml` | 115.30% | 4.175 | -9.65% | 3.35 | 86% |
| `hf_concurrent3_105pct.yaml` | 105.44% | 3.930 | -9.98% | 3.45 | 100% |
| `hf_highret_122pct.yaml` | 122.86% | 3.512 | -12.17% | 1.29 | 100% |

## Architecture: regime_specialized + scalp_momentum
Mode: `strategies.mode: regime_specialized` — two sub-strategies routed by HMM regime probability:
1. **trend**: 10-bar Donchian Channel breakout. SL=2.0x ATR, TP=3.0x ATR.
2. **scalp_momentum**: 8-bar DC breakout on 5M. SL=1.0x ATR, TP=1.5x ATR. Most of the trade frequency.

ATR expansion is a REQUIRED GATE — ATR must be above 20-bar rolling mean to enter.
The sideways regime generates zero signals in practice (prob < 0.05 on 97%+ of bars).

## Multi-Position Concurrent Engine
- File: `code3.0/src/trade2/backtesting/engine.py`
- Function: `_simulate_trades_multi()`
- Activated by: `risk.max_concurrent_positions: N` (N > 1)
- Each position gets `base_allocation_frac / max_concurrent` capital — total risk unchanged
- Signal queue: one pending entry at a time; discarded if all slots full (conservative)
- Default: `max_concurrent_positions: 1` (backward-compatible)

## Planning Workflow
1. Create a plan file in `.claude/steps/` before writing any code.
   - Format: `YYYY-MM-DD_short_description.md`
   - Contents: goal, file list, step-by-step notes, verification steps.
2. Show plan to user and wait for approval before implementing.
3. Append `## Completed` section with date + commit hash when done.

## Live Trading Module
Single Python process, 24/5 polling loop (10s), two strategies simultaneously.

```bash
trade2-live                     # run both strategies (A=magic 100001, B=magic 100002)
trade2-live --report            # performance report and exit
trade2-live --retrain           # force HMM retrain
```

Credentials in `code3.0/.env`: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER=Exness-MT5Trial`
Config: `code3.0/configs/live.yaml`
Live model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl` (36 features)
Weekly retrain: every Sunday on original 2019-2025 data + accumulated live bars.

## Scalping Research Loop
```bash
scalp_research --source seed --max-ideas 20    # use curated seed list
scalp_research --source llm --trials 30        # LLM discovery
scalp_research --dry-run --max-ideas 2         # translate only
```
Goals: return>=30%, sharpe>=1.5, dd>=-20%, tpd>=5
Key files: `app/scalp_research_loop.py`, `configs/scalp.yaml`, `configs/scalp_seed_list.yaml`

## TV Indicator Research Loop
```bash
tv_research --max-ideas 5 --source seed --trials 50
tv_research --dry-run --max-ideas 3
```
Goals: return>=50%, sharpe>=1.5, dd>=-25%
Modules saved to: `code3.0/src/trade2/features/tv_indicators/{name}.py`

## Config Rules
- Config is the SINGLE SOURCE OF TRUTH — no hardcoded fallback values in Python
- Use `cfg["section"]["key"]` directly, never `.get()` with hardcoded defaults
- If a key is missing, code should fail loudly (KeyError)

## Key Module Paths
- `trade2/config/loader.py` — load_config(base, override) with deep merge
- `trade2/features/builder.py` — feature construction
- `trade2/features/hmm_features.py` — HMM feature set
- `trade2/features/smc_luxalgo.py` — SMC zones (BOS/CHoCH, OB, premium/discount)
- `trade2/backtesting/engine.py` — single + concurrent + partial-TP engines
- `trade2/signals/generator.py` — signal routing
- `trade2/app/run_pipeline.py` — `trade2` CLI entry point
- `configs/base.yaml` — single source of truth for all parameters

## BTCUSDT Research (isolation rule)
- NEVER modify XAUUSD configs/code/approved folders for BTC work
- BTC configs: `code3.0/configs/btcusdt/`
- BTC artefacts: `code3.0/artefacts/btcusdt/`
- BTC HMM: always retrain on BTC data (never reuse XAUUSD golden model)
- Data: `code3.0/data/raw/btcusdt/BTCUSDT_{1H,5M,...}_2017_2026.csv`
- Run: `trade2 --base-config configs/btcusdt/base_btcusdt.yaml --config configs/btcusdt/<strategy>.yaml --retrain-model`

## Experiment Archive
Full results log (all runs, good and bad): `code3.0/configs/experiments/RESULTS.md`
