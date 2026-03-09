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
      engine.py                   # vectorbt backtest runner
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

## User Preferences
- Auto-proceed without asking for confirmation
- No unicode special characters in print statements (cp874 encoding)
- Vectorbt is the primary backtesting framework
- Save strategies achieving >= 50% return
