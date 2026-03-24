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

This directory contains two live-deployed strategies and is permanently frozen:

| Strategy | Folder | Test Return | Sharpe | DD |
|----------|--------|-------------|--------|----|
| 89% (Strategy A) | `xauusd_mtf_hmm1h_smc5m_2026_03_18/` | 89.33% | 4.10 | -7.53% |
| 49% (Strategy B) | `xauusd_mtf_hmm1h_smc5m_tp2x_49pct_2026_03_18/` | 49.27% | 3.16 | -7.15% |

Rules:
- Do NOT edit `config.yaml`, `metrics.json`, `trades_test.csv`, or `training_summary.md` in either folder.
- Do NOT delete or overwrite `model.pkl` in either folder.
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

## CLAUDE.md Maintenance
CLAUDE.md must be kept up to date at all times. After any session where new modules, results, architecture decisions, configs, CLI usage patterns, or user preferences are introduced or changed, update the relevant section(s) of CLAUDE.md before finishing.

- Update **Repository Structure** when new files/folders are added
- Update **User Preferences** when new preferences are established
- Update **Quick Start / CLI Usage** when entry points or flags change
- Add new sections as needed for major features (e.g., new agents, strategies, dashboards)
- Remove stale information that no longer reflects the codebase
