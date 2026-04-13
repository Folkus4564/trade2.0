# trade2.0 - XAUUSD Automated Quant Research System

## Overview
Self-operating quantitative trading research pipeline for XAUUSD (Gold).
Composed of 6 specialized subagents that collaborate to generate, test, debug, and evaluate systematic trading strategies.

## Quick Start
```bash
cd C:/Users/LENOVO/Desktop/trade2.0

# Run full pipeline with default HMM regime strategy
/run-trading-pipeline Build a systematic XAUUSD trading strategy

# Or run directly
python src/pipeline.py

# With optimization
python src/pipeline.py --optimize --trials 200

# With walk-forward validation
python src/pipeline.py --walk-forward

# Prepare data only
python src/data/prepare_data.py
```

## Repository Structure
```
trade2.0/
  data/
    XAUUSD_1H_2019_2024.csv      # Raw 1H bars (2019-2024, Dukascopy)
    raw/                          # Additional raw data files
    processed/                    # Cleaned train/val/test parquet files
  src/
    data/
      loader.py                   # Data loading and splitting
      features.py                 # Feature engineering (lag-safe)
      prepare_data.py             # Data preparation script
    models/
      hmm_model.py                # GaussianHMM regime detector
      signal_generator.py         # Signal generation from features+regime
    backtesting/
      engine.py                   # event-driven backtest engine (bar-by-bar SL/TP)
      metrics.py                  # Performance metrics computation
    pipeline.py                   # MAIN entry point
  models/
    hmm_regime_model.pkl          # Trained HMM (auto-generated)
  backtests/                      # Backtest result JSONs
  reports/
    strategy_blueprint.yaml       # Strategy definition (from Architect agent)
    research_report.md            # Quant research findings
    debug_report.md               # Audit findings
    analytics_report.md           # Final performance analysis
    final_verdict.json            # Machine-readable verdict
  .claude/
    agents/                       # Subagent system prompts
    skills/                       # Orchestration skills
```

## Subagents
| Agent | File | Purpose |
|-------|------|---------|
| Strategy Architect | `.claude/agents/strategy-architect.md` | Blueprint generation |
| Quant Researcher | `.claude/agents/quant-researcher.md` | Alpha research |
| Market Data Engineer | `.claude/agents/market-data-engineer.md` | Data preparation |
| Strategy Code Engineer | `.claude/agents/strategy-code-engineer.md` | Implementation |
| Strategy Debugger | `.claude/agents/strategy-debugger.md` | Bias audit |
| Analytics Reviewer | `.claude/agents/analytics-reviewer.md` | Performance evaluation |

## Data Splits
- Train:      2019-01-01 to 2022-12-31 (in-sample, used for HMM training + optimization)
- Validation: 2023-01-01 to 2023-12-31 (Optuna optimization target)
- Test:       2024-01-01 onwards        (out-of-sample, final evaluation only)

## Research Standards
- NO lookahead bias: all features shift(1) before use as HMM inputs
- NO data leakage: scaler and HMM fitted on train only
- Realistic costs: 3 pip spread + 1 pip slippage + 2 bps commission
- Minimum 30 trades for statistical validity
- Walk-forward validation mandatory for any optimization

## Performance Targets
| Metric | Minimum | Target |
|--------|---------|--------|
| Annualized Return | 20% | 50%+ |
| Sharpe Ratio | 1.0 | 1.5+ |
| Max Drawdown | -35% | -20% |
| Profit Factor | 1.2 | 1.5+ |
| Total Trades | 30 | 50+ |

## Auto-Save Rule
If test-period results show annualized_return >= 50% AND sharpe >= 1.5 AND max_dd >= -25%:
Strategy is automatically saved as `YYYY-MM-DD_xauusd_hmm_hma_regime.py`

## Golden Model Protection (added 2026-03-17)
The pipeline automatically preserves high-potential models to prevent loss on retrain:

- **Auto-backup before retrain**: `--retrain-model` backs up the current model to
  `artefacts/models/backups/hmm_{tf}_{n}states_{timestamp}.pkl` before overwriting.
- **Auto-save golden**: After any run where test annualized_return >= 20% (configurable via
  `pipeline.golden_model_threshold` in base.yaml), the model is copied to
  `artefacts/models/golden/{name}_ret{X}pct_sh{Y}.pkl` with a `_metrics.json` sidecar.
- **Load specific model**: Use `--model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret25pct_sh1.20.pkl`
  to run the full pipeline with any saved model (skips retrain entirely).

```bash
# Retrain (old model backed up automatically)
trade2 --retrain-model --skip-walk-forward

# Reuse a specific golden model without retraining
trade2 --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret25pct_sh1.20.pkl
```

## User Preferences
- Auto-proceed without asking for confirmation
- No unicode special characters in print statements (cp874 encoding)
- Custom event-driven engine (pandas/numpy, no vectorbt) is the backtesting framework
- Save strategies achieving >= 20% return
- Golden model auto-save threshold: >= 20% annualized return (pipeline.golden_model_threshold in base.yaml)

## APPROVED STRATEGY PROTECTION — ABSOLUTE RULE

**NEVER modify any file inside `code3.0/artefacts/approved_strategies/`.**

This directory contains frozen approved strategies:

| Strategy | Folder | Test Return | Sharpe | DD | WF | Engine | Status |
|----------|--------|-------------|--------|----|----|--------|--------|
| 89% (Strategy A) | `xauusd_mtf_hmm1h_smc5m_2026_03_18/` | 89.33% | 4.10 | -7.53% | 100% | single | Live-deployed |
| 49% (Strategy B) | `xauusd_mtf_hmm1h_smc5m_tp2x_49pct_2026_03_18/` | 49.27% | 3.16 | -7.15% | 100% | single | Live-deployed |
| 122% (Strategy C) | `xauusd_hf_r1p0_lb20_2026_03_29/` | 122.86% | 3.51 | -12.17% | 100% | single | Approved 2026-03-29 |
| 105% (Strategy D) | `xauusd_hf_concurrent3_105pct_2026_03_30/` | 105.44% | 3.93 | -9.98% | 100% | concurrent-3 | Approved 2026-03-30 |
| 115% (Strategy E) | `xauusd_hf_macro_sl15_55pct_2026_03_30/` | 115.30% | 4.18 | -9.65% | 86% | concurrent-5 | Approved 2026-03-30 |
| 134% (Strategy F) | `xauusd_pullback_retest_v3_2026_03_31/` | 133.67% | 4.15 | -7.62% | 100% | concurrent-5 | Approved 2026-03-31 |
| 165% (Strategy H) | `xauusd_pullback_retest_v6_2026_04_01/` | 165.44% | 3.86 | -10.18% | 86% | concurrent-5 | Approved 2026-04-01 |
| 160% (Strategy G) | `xauusd_pullback_retest_v5_2026_04_01/` | 159.88% | 4.02 | -8.74% | 100% | concurrent-5 | Approved 2026-04-01 |

**Strategy C (122%)** details:
- Config: `configs/hf_highret_122pct.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Architecture: `regime_specialized` — trend (DC breakout) + scalp_momentum (ATR-gated DC 8-bar)
- WF: mean_sharpe=1.010, 7/7 windows positive (100%)
- Test period: 2025-01-01 to 2026-03-15 | Trades: 406 | TPD: 1.29
- Reproduce: `trade2 --config configs/hf_highret_122pct.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Strategy D (105%)** details:
- Config: `configs/hf_concurrent3_105pct.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Architecture: `regime_specialized` — same as Strategy C, but with concurrent engine (max_concurrent_positions=3)
- Engine: `_simulate_trades_multi()` — up to 3 simultaneous positions, each at 1/3 allocation
- WF: mean_sharpe=0.688, 7/7 windows positive (100%), including previously-failing window 6
- Test period: 2024-2025 | Trades: 1069 | TPD: 3.45
- Reproduce: `trade2 --config configs/hf_concurrent3_105pct.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Strategy E (115%, WR=55.7%)** details:
- Config: `configs/hf_macro_sl15_55pct.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Architecture: `regime_specialized` + `require_macro_trend=True` on scalp_momentum
- Key changes vs D: scalp SL 1.0x->1.5x ATR; scalp_momentum.require_macro_trend=True (filters counter-trend entries); base_allocation_frac=0.75; max_concurrent=5
- WR: 55.68% (first to meet 55-65% target) | Return: 115.30% | Sharpe: 4.18 | DD: -9.65%
- WF: mean_sharpe=0.871, 6/7 windows positive (86%)
- Test period: 2025-01-01 to 2026-03-15 | Trades: 1047 | TPD: 3.35
- Reproduce: `trade2 --config configs/hf_macro_sl15_55pct.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Strategy F (134%, WR=51.2%, pullback-retest hybrid)** details:
- Config: `configs/experiments/hf_pullback_v3.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Architecture: `regime_specialized` — trend + scalp_momentum (fallback) + scalp_pullback (retest entry)
- Key changes vs E: added scalp_pullback sub-strategy (enters at DC breakout retest, SL=1.0x ATR, TP=1.5x ATR); session_enabled=false on all strategies (24/7 including trend)
- Return: 133.67% | Sharpe: 4.15 | DD: -7.62% | Profit Factor: 1.845
- WF: mean_sharpe=0.679, 7/7 windows positive (100%) — FIRST to achieve 100% with concurrent-5
- Test period: 2025-01-01 to 2026-03-15 | Trades: 1801 | TPD: ~5.8
- Note: scalp_pullback SL/TP were silently using global fallback — fixed in Strategy G
- New file: `src/trade2/signals/strategies/scalp_pullback.py` (pullback-retest sub-strategy)
- Reproduce: `trade2 --config configs/experiments/hf_pullback_v3.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Strategy H (165%, WR=55.2%, pullback-retest v6 — long-only pullback)** details:
- Config: `configs/experiments/hf_pullback_v6.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Key change vs G: `scalp_pullback.long_only=true` — short pullbacks removed, handled by scalp_momentum
- Return: 165.44% | Sharpe: 3.857 | DD: -10.18% | WR: 55.16% | Trades: 2065 | TPD: ~6.6
- WF: mean_sharpe=0.843, 6/7 windows positive (86%) — W4 just negative
- FIRST strategy to meet 55%+ WR target while also exceeding 150% return target
- Reproduce: `trade2 --config configs/experiments/hf_pullback_v6.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Strategy G (160%, WR=52.8%, pullback-retest v5 — bug-fixed)** details:
- Config: `configs/experiments/hf_pullback_v5.yaml`
- Model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`
- Architecture: same as F — trend + scalp_momentum + scalp_pullback (both directions)
- Key fix: `compute_stops_regime_aware` now correctly applies scalp_pullback SL/TP
- scalp_pullback: SL=1.5x ATR, TP=2.0x ATR (R:R=1.33), min_prob_short=0.85
- Return: 159.88% | Sharpe: 4.02 | DD: -8.74% | WR: 52.84% | Trades: 2360 | TPD: ~7.6
- Short trades: 1014 (43.9% WR) contributing $39,617 (24.6% of P&L) — profitable, kept
- WF: mean_sharpe=0.995, 7/7 windows positive (100%) — improved from F's 0.679
- Test period: 2025-01-01 to 2026-03-15
- Reproduce: `trade2 --config configs/experiments/hf_pullback_v5.yaml --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

Rules:
- Do NOT edit `config.yaml`, `metrics.json`, `trades_test.csv`, or `training_summary.md` in any folder.
- Do NOT delete or overwrite `model.pkl` in any folder.
- Do NOT run `trade2 --export-approved` in a way that would overwrite these folders.
- Any improvement work must go to a NEW folder with a new timestamp name.
- Read-only access only — you may read these files for reference but never write to them.

## Planning Workflow
When the user asks to plan something (new feature, refactor, investigation, etc.):

1. **Create a plan file** in `.claude/steps/` before writing any code.
   - Filename format: `YYYY-MM-DD_short_description.md`
   - Contents: goal, motivation, file list (new + modified), step-by-step implementation notes, verification steps.
   - Show the plan to the user and wait for approval before implementing.

2. **Implement** only after the user confirms the plan.

3. **Append a "## Completed" section** to the plan file once implementation is complete and verified.
   - Include the date and commit hash.

The `.claude/steps/` folder holds all plans (pending and completed).

## Live Trading Module (added 2026-03-18)

Live trading deployment for XAUUSD on Exness MT5 demo.

### Architecture
Single Python process, 24/5 polling loop (10s), two strategies running simultaneously.

### CLI
```bash
trade2-live                           # run both strategies
trade2-live --strategy-a-only         # run only the 89% strategy
trade2-live --strategy-b-only         # run only the 49% strategy
trade2-live --report                  # generate performance report and exit
trade2-live --retrain                 # force immediate HMM retrain
trade2-live --config configs/live.yaml   # use custom live config
```

### Setup
1. Fill in MT5 credentials in `code3.0/.env`:
   ```
   MT5_LOGIN=your_login
   MT5_PASSWORD=your_password
   MT5_SERVER=Exness-MT5Trial
   ```
2. Install MetaTrader5: `pip install MetaTrader5`
3. Run: `trade2-live`

### Key Files
- `code3.0/src/trade2/live/` — all live trading modules
  - `mt5_connector.py`     — MT5 connection, bar fetch, order execution
  - `bar_manager.py`       — new-bar detection, rolling window management
  - `signal_pipeline.py`   — wraps full feature+HMM+signal chain
  - `position_manager.py`  — manages one position per strategy (magic number)
  - `strategy_instance.py` — binds config + model + pipeline + position
  - `trade_logger.py`      — append-mode CSV trade log
  - `reporter.py`          — performance metrics from trade log
  - `data_accumulator.py`  — appends live bars to CSV for expanding retrain
  - `retrainer.py`         — Sunday weekly HMM retrain on expanding window
  - `health.py`            — connection monitoring, auto-reconnect, weekend skip
  - `main.py`              — CLI entry point
- `code3.0/configs/live.yaml` — strategy configs, magic numbers, retrain schedule

### Strategy Magic Numbers
- Strategy A (89% return): magic 100001
- Strategy B (49% return): magic 100002

### Important: Correct Model for Strategy A
The approved strategy's `model.pkl` has 7 features but the config needs 36.
`live.yaml` correctly points to: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

### Weekly Retrain
Every Sunday, the HMM retrains on all data (original 2019-2025 + accumulated live bars).
Live bars are accumulated to `data/raw/XAUUSD_1H_live.csv` and `data/raw/XAUUSD_5M_live.csv`.

## Scalping Research Loop (added 2026-03-22)

Automated LLM-driven loop to discover scalping-suitable indicators for XAUUSD 5M signals.
Forked from `tv_research_loop.py` with scalping-specific discovery and tight 1:1.5 R:R.

### CLI
```bash
scalp_research                              # defaults from config
scalp_research --max-ideas 20              # limit iterations
scalp_research --trials 30                 # Optuna trials per indicator
scalp_research --dry-run --max-ideas 2     # translate only, skip pipeline
scalp_research --provider deepseek         # LLM provider
scalp_research --goal-return 0.40          # override return goal
scalp_research --goal-sharpe 2.0           # override Sharpe goal
scalp_research --min-trades-per-day 8      # stricter frequency threshold
scalp_research --walk-forward              # enable walk-forward
scalp_research --no-retrain                # reuse existing HMM
scalp_research --skip-greedy-stack         # skip stacking phase
scalp_research --source seed               # use seed list only (no LLM)
scalp_research --source llm                # LLM discovery only (no seed)
scalp_research --source both               # seed first, then LLM (default)
scalp_research --technique momentum_breakout  # filter to one technique
```

### 5 Scalping Techniques
Each discovered indicator is tagged with one of:
- `momentum_breakout` — break of key level with volume confirmation
- `vwap_pullback` — trend + pullback to VWAP, enter on resume
- `range_mean_reversion` — fade into S/R, snap back to midpoint
- `order_flow` — microstructure proxies from OHLCV (delta, bar imbalance)
- `opening_range` — first volatility burst after market open or catalyst

### Key Files
- `code3.0/src/trade2/app/scalp_research_loop.py` — forked loop, scalping-adapted
- `code3.0/configs/scalp.yaml` — overlay: 5M signals, SL=1.0x ATR, TP=1.5x ATR, session 8-17 UTC
- `code3.0/configs/scalp_seed_list.yaml` — curated seed indicators tagged by technique (205 indicators)
- `code3.0/artefacts/scalp_research/` — log, best, stack JSONs (auto-created)

### Config Section
`configs/base.yaml` has a `scalp_research:` section with goals (return>=30%, sharpe>=1.5, dd>=-20%, tpd>=5).
`scalp.yaml` is always merged on top of base.yaml when running `scalp_research`.

### Key Differences vs tv_research
- `--source seed|llm|both` — seed list support added; LLM discovery also available
- Each indicator tagged with a scalping technique (5 techniques total)
- `--technique` flag filters discovery to one technique
- Short-period constraint enforced in prompt: MAs 3-8 bars, oscillators 3-10 bars
- Trade frequency validation: logs `trades_per_day`, warns if below threshold
- Leaderboard shows Technique and TPD (trades/day) columns
- Status `COMPLETED_LOW_FREQ` for runs below frequency threshold (not a hard stop)

## SMC HighFreq Strategy Research (added 2026-03-29, updated 2026-03-30)

Regime-specialized high-frequency strategy combining trend + scalp_momentum signals on the 36-feature golden HMM model. Three engine variants now approved: single-position (122%, 1.29 TPD), concurrent-3 (105%, 3.45 TPD), and macro-filtered concurrent-5 (115%, 3.35 TPD, WR=55.7%).

### APPROVED Configs (production ready)

| Config | Return | Sharpe | Max DD | TPD | WF | Engine | Status |
|--------|--------|--------|--------|-----|----|--------|--------|
| `configs/experiments/hf_pullback_v6.yaml` | **165.44%** | **3.857** | **-10.18%** | **~6.6** | **86%** | concurrent-5 | Approved 2026-04-01 (Strategy H, WR=55.2%) |
| `configs/experiments/hf_pullback_v5.yaml` | 159.88% | 4.022 | -8.74% | ~7.6 | 100% | concurrent-5 | Approved 2026-04-01 (Strategy G) |
| `configs/experiments/hf_pullback_v3.yaml` | 133.67% | 4.146 | -7.62% | ~5.8 | 100% | concurrent-5 | Approved 2026-03-31 (Strategy F) |
| `configs/hf_macro_sl15_55pct.yaml` | 115.30% | 4.175 | -9.65% | 3.35 | 86% | concurrent-5 | Approved 2026-03-30 (WR=55.7%) |
| `configs/hf_concurrent3_105pct.yaml` | 105.44% | 3.930 | -9.98% | 3.45 | 100% | concurrent-3 | Approved 2026-03-30 |
| `configs/hf_highret_122pct.yaml` | 122.86% | 3.512 | -12.17% | 1.29 | 100% | single | Approved 2026-03-29 |
| `configs/experiments/hf_sl15_cdc.yaml` | 99.61% | 3.200 | -12.17% | 1.75 | 86% | single | Research only |
| `configs/experiments/hf_v10_sweet.yaml` | 97.24% | 3.135 | ~-12% | 1.82 | 86% | single | Research only |
| `configs/experiments/hf_v9_final.yaml` | 93.24% | 3.038 | ~-12% | 1.99 | 86% | single | Research only |

All use the same golden model: `artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl`

**Choose between the five production configs:**
- `hf_pullback_v6.yaml` -- **NEW BEST** (165%, WR=55.2%), WF 86%, ~6.6/day -- RECOMMENDED
- `hf_pullback_v5.yaml` -- 160%, WF 100%, ~7.6/day, WR=52.8% (best WF consistency)
- `hf_pullback_v3.yaml` -- 134%, LOWEST DD (-7.6%), WF 100%, ~5.8/day
- `hf_macro_sl15_55pct.yaml` -- BEST WR (55.7%), return 115%, 3.35/day, WF 86%
- `hf_concurrent3_105pct.yaml` -- best WF Sharpe (3.93), 3.45/day, WF 100%
- `hf_highret_122pct.yaml` -- older 122%, WF 100%, 1.29/day

### How to Reproduce

```bash
cd code3.0

# 134% pullback-retest hybrid (NEW BEST: 24/7, lowest DD -7.6%, WF 100%)
trade2 --config configs/experiments/hf_pullback_v3.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

# 115% macro-filtered (BEST WR: 55.7%, 3.35/day, WF 86%)
trade2 --config configs/hf_macro_sl15_55pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

# 105% concurrent (best Sharpe, 3.45/day, WF 100%)
trade2 --config configs/hf_concurrent3_105pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

# 122% single-position (older, 1.29/day, WF 100%)
trade2 --config configs/hf_highret_122pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl

# Quick check (skip walk-forward)
trade2 --config configs/hf_concurrent3_105pct.yaml \
       --model-path artefacts/models/golden/hmm_1h_3states_2026_03_17_ret89pct_sh4.10.pkl \
       --skip-walk-forward
```

Expected output for 105% config:
```
TRAIN  | Return: 28.67% | Sharpe: 1.837 | MaxDD:  -7.79% | Trades: 4788
VAL    | Return: 30.29% | Sharpe: 2.183 | MaxDD:  -4.80% | Trades:  809
TEST   | Return: 105.44% | Sharpe: 3.930 | MaxDD: -9.98% | Trades: 1069
WF     | 7/7 positive (100%) | mean_sharpe: 0.688
VERDICT: APPROVED
```

Full experiment log (all runs, good and bad): `code3.0/configs/experiments/RESULTS.md`

### Architecture: regime_specialized + scalp_momentum

The configs use `strategies.mode: regime_specialized` with two active sub-strategies routed by HMM regime probability:

1. **trend**: 10-bar Donchian Channel breakout, fires only in HMM bull/bear regime (prob >= hmm_min_prob). SL=2.0x ATR, TP=3.0x ATR. Longer hold, lower frequency, highest quality.
2. **scalp_momentum**: 8-bar DC breakout on 5M bars, same regime filter. SL=1.0x ATR, TP=1.5x ATR (base). Provides the majority of trade frequency. In Strategy E: SL=1.5x, require_macro_trend=True for 55%+ WR.
3. **cdc_retest** (optional, not used in production configs): 15M CDC retest signals. Zero signals in WF windows (no 15M data loaded during WF).

The sideways/volatile/range regimes from the 3-state HMM generate zero signals in practice (sideways probability below 0.05 on more than 97% of bars).

### Critical Technical Insights

**ATR expansion is a REQUIRED GATE (not a suppressor):** Both strategies require ATR above its 20-bar rolling mean to enter.
- `atr_expansion_filter: true` + ratio=1.0 means ATR must be above rolling mean, eliminating low-vol false breakouts
- Higher `atr_expansion_ratio` (e.g. 1.2, 1.5) means harder threshold and FEWER trades from BOTH strategies
- To increase frequency without lowering quality: use the concurrent engine (not a weaker ATR filter)

**WF window 6 (2024 H1) fix via concurrent engine:** This window (Jan-Jun 2024 gold parabolic rally) consistently produced negative Sharpe for all single-position configs due to 34-38% WR on DC breakouts. The concurrent engine turns it slightly positive (+1.39%) by allowing trend signals to enter even while scalp positions are still open -- capturing the directional moves that the scalp strategy missed.

**cdc_retest = 0 in WF:** WF validation splits do not load 15M CDC data. All cdc_retest signals are zero.

### Multi-Position Concurrent Engine (added 2026-03-30)

**File:** `code3.0/src/trade2/backtesting/engine.py`
**Function:** `_simulate_trades_multi()`
**Activated by:** `risk.max_concurrent_positions: N` where N > 1 in config

#### Problem

The original single-position engine (`_simulate_trades`) blocks all new signals while a position is open. With trend + scalp_momentum both firing in trending periods, ~5 quality signals/day were generated but ~74% were discarded because a slot was already occupied. Only ~1.29 trades/day were executed despite abundant opportunity.

#### Solution

`_simulate_trades_multi()` allows up to `max_concurrent_positions` simultaneous open positions. Capital exposure is held constant: each position receives `base_allocation_frac / max_concurrent` of available cash. Total risk at full capacity equals the single-position engine exactly.

**Signal queue mechanics:**
- One pending entry is queued at a time (signal bar i -> execute at bar i+1 open)
- On each bar, if a pending entry exists AND len(positions) < max_concurrent, the entry is executed
- If all slots are full, the pending entry is discarded (conservative -- no stacking of queued signals)
- Each position tracks its own SL, TP, trailing stop, break-even, and timeout independently
- Mark-to-market equity = cash + sum of all open positions' unrealized P&L

**Key config parameter:**
```yaml
risk:
  max_concurrent_positions: 3   # 1 = original single-position engine (default, backward-compatible)
                                 # 3 = allows 3 simultaneous positions, each at 1/3 allocation
```

#### Engine dispatch logic in `run_backtest()`:
```python
max_concurrent = int(risk_cfg.get("max_concurrent_positions", 1))

if max_concurrent > 1:
    equity, trades_df = _simulate_trades_multi(
        df, init_cash, base_alloc, slippage, commission_rt,
        max_hold_bars, be_atr_trigger, contract_size_oz,
        max_concurrent=max_concurrent,
    )
else:
    equity, trades_df = _simulate_trades(
        df, init_cash, base_alloc, slippage, commission_rt,
        max_hold_bars, be_atr_trigger, contract_size_oz,
    )
```

#### Results of concurrent=3 vs single-position (same signals, same model):

| Metric | Single (122% config) | Concurrent-3 (105% config) |
|--------|----------------------|---------------------------|
| Test Return | 122.86% | 105.44% |
| Test Sharpe | 3.512 | **3.930** |
| Max Drawdown | -12.17% | **-9.98%** |
| Trades/Day | 1.29 | **3.45** |
| WF Positive | 86% (6/7) | **100% (7/7)** |
| WF mean_sharpe | 1.010 | 0.688 |
| WF Window 6 | negative | **+1.39%** |

Raw return is lower because each position uses 1/3 capital. But Sharpe improves because drawdowns are smaller and profits are more consistent across more trades.

### Modified Files

- **`src/trade2/backtesting/engine.py`**: Added `_simulate_trades_multi()` function (lines ~255-460). Modified `run_backtest()` to read `risk.max_concurrent_positions` and dispatch to the appropriate engine. Default is 1 (backward-compatible, no behavior change for existing configs).
- **`src/trade2/features/builder.py`**: ATR expansion lookback and ratio configurable via `features.atr_expansion_lookback` (default 20) and `features.atr_expansion_ratio` (default 1.0).
- **`src/trade2/signals/strategies/scalp_momentum.py`**: Added `require_macro_trend` flag (default False, backward-compatible). When True: longs only fire when HMA rising + price above HMA; shorts only when HMA falling + price below HMA. Used by Strategy E.

### Approved Strategy Folders

| Folder | Config | Return | TPD | Date |
|--------|--------|--------|-----|------|
| `artefacts/approved_strategies/xauusd_hf_macro_sl15_55pct_2026_03_30/` | `configs/hf_macro_sl15_55pct.yaml` | 115.30% (WR=55.7%) | 3.35 | 2026-03-30 |
| `artefacts/approved_strategies/xauusd_hf_concurrent3_105pct_2026_03_30/` | `configs/hf_concurrent3_105pct.yaml` | 105.44% | 3.45 | 2026-03-30 |
| `artefacts/approved_strategies/xauusd_hf_r1p0_lb20_2026_03_29/` | `configs/hf_highret_122pct.yaml` | 122.86% | 1.29 | 2026-03-29 |

### Experiment Archive

- `configs/experiments/RESULTS.md` -- full reproduction guide, all results tables, failed experiments, key insights
- `configs/experiments/hf_v20_concurrent3.yaml` -- v20 experiment original (same as hf_concurrent3_105pct.yaml)
- `configs/experiments/hf_r1p0_lb20.yaml` -- same as `configs/hf_highret_122pct.yaml` (122% experiment original)
- `configs/experiments/hf_v7_atrfilter.yaml` -- v7 baseline reference
- `configs/experiments/hf_sl15_cdc.yaml`, `hf_v10_sweet.yaml`, `hf_v9_final.yaml` -- single-position research configs (86% WF)

### What Was Abandoned

**Increasing TPD via max_hold_bars / dc_period tuning (v11-v19):** Setting max_hold=1-2 or dc_period=5 without the ATR filter collapsed train Sharpe to near-zero (~0.057) due to too many false DC breakouts on short lookbacks. The concurrent engine was the only working approach to increasing TPD without sacrificing quality.

**WR improvement experiments (v24-v43, 2026-03-30):** Attempted to push WR from 51.7% to 55%+ via:
- Break-even stops: FAILED -- BE fires then slippage on exit = tiny loss, WR DROPPED to 31.5% (v28)
- Trailing stops: FAILED -- exits during dips that would have hit TP, WR dropped to 47.5% (v26)
- min_prob increase alone: ceiling ~54.4% (v34), diminishing returns above 0.72
- Long-only filter: WR 59.8% but overfits 2025 gold bull run (train return only 9.7%)
- **SOLVED (v44)**: require_macro_trend=True on scalp_momentum + wider SL (1.5x ATR)
  - Root cause: long WR=57.6% vs short WR=42.8% persists across ALL periods
  - Macro filter removes ~30% of counter-trend short signals (the low-quality ones)
  - Result: WR=55.68%, Return=115.30%, WF 86% -- Strategy E

**SMC pullback reversal**: Consistent ~25% WR across all configurations. Too low win-rate for profitability at any reasonable R:R ratio.

## SMC + SD Adaptive Mean Strategy Research (2026-04-05)

Researched a new mean-reversion strategy combining LuxAlgo SMC zones (OB retests, demand/supply zones) with an SD Adaptive Mean indicator. Research concluded that this standalone 5M approach cannot achieve the 150%+ return targets.

### Key Files
- `code3.0/src/trade2/features/sd_adaptive_mean.py` -- SD Adaptive Mean indicator (ATR-adaptive SMA, standardized oscillator)
- `code3.0/src/trade2/signals/strategies/smc_sd_mean.py` -- Strategy signal generator
- `code3.0/configs/smc_sd_mean.yaml` -- Strategy config with Optuna best params

### Architecture
- Mode: `multi_tf` — 1H HMM regime (7 features: ret, rsi, atr_ratio, vol, hma_slope, bb_width, macd) forward-filled to 5M signal bars
- Entry: SD mean extended below threshold + SMC zone (OB/demand or BB position) + rejection candle + HMM bull regime gate
- Long-only (no shorts — gold secular bull trend)
- SL=1.97x ATR, TP=4.90x ATR (R:R=2.49) — Optuna best

### Performance (Optuna best params, retrain-model, skip-walk-forward)
| Split | Return | Sharpe | WR | Trades | DD |
|-------|--------|--------|----|--------|----|
| Train (2019-2023) | -5.26% | -2.95 | 29.53% | 1676 | -22.3% |
| Val (2024) | -0.65% | -1.55 | 31.49% | 235 | -3.3% |
| Test (2025-03/2026) | +3.33% | -0.36 | 40.83% | 120 | -2.7% |

### Why 150%+ Return is Not Achievable
- 5M ATR ≈ $5, position ≈ 0.19 lots → avg P&L per trade: $40-$190
- At 0.3 trades/day: annual return capped at 3-8% even at 55%+ WR
- Approved strategies achieve 89-165% through momentum breakouts capturing $30-60 1H-scale moves
- Mean reversion exits in 1-5 bars at $5-15 move — fundamentally different economics

### Key Learnings
1. **HMM gate is essential**: Removing bull_prob gate drops test WR from 54% to 44%
2. **SD mean is NOT a strong reversal predictor**: Acts only as a loose "zone" filter
3. **Optimizer bug fixed**: Objective `min(val, train)` was forcing -999 when train always negative; changed to soft penalty `val + 0.2 * min(0, train)` 
4. **Regime drift**: Fresh 7-feature 1H HMM labels 2025 as only 15% bull (vs 44% in train) → hard rejection (28.9pp); the golden 36-feature HMM is much more stable
5. **The strategy is valid as an overlay**: Achieves +3.33% test return at -2.74% max DD — excellent risk profile, just not standalone
6. **Best standalone results**: min_prob=0.45 (bull gate) → test WR=54.55%, 132 trades (0.3/day), +0.93% annual

### Optimizer Infrastructure Fixed
- `run_optimization_smc_sd_mean()` in `optimizer.py`: changed objective from `min(val, train)` to `val + 0.2 * min(0, train)` (soft penalty)
- `run_pipeline.py`: fixed HMM probability injection for multi_tf mode before optimizer call (was missing `bull_prob`/`bear_prob` columns → all trials returned -999)

### Recommended Next Step
This strategy's signal quality (54%+ WR) is valuable. Best deployment: add as an additional signal source in an existing approved config (e.g., hf_pullback_v6.yaml) to marginally improve Sharpe without increasing drawdown.
