# Live Trading Deployment — XAUUSD on Exness MT5

**Date:** 2026-03-18
**Status:** Draft

## Problem

The backtesting pipeline has produced 2 APPROVED strategies with strong out-of-sample results (89% and 49% annualized return). There is no way to deploy these strategies to execute real trades. The user wants to move from research to live execution on Exness via MetaTrader5.

## Goals

1. Deploy 2 approved strategies independently on Exness MT5 (demo first)
2. Match backtest position sizing exactly
3. Log every live trade with full context
4. Generate performance reports comparing live vs backtest
5. Store MT5 credentials securely in `.env`

## Architecture

**Single-process polling loop** running 24/5 on Windows. Polls MT5 every 10 seconds, detects 5M bar closes, runs the full signal pipeline (reusing all existing feature/HMM/signal code), and executes orders via MT5 Python API.

### New Modules (`code3.0/src/trade2/live/`)

| Module | Purpose |
|--------|---------|
| `mt5_connector.py` | MT5 connection, bar fetching, order execution |
| `bar_manager.py` | New-bar detection, DataFrame conversion |
| `signal_pipeline.py` | Wraps existing feature -> HMM -> signal chain |
| `position_manager.py` | Maps signals to MT5 orders per strategy |
| `strategy_instance.py` | Binds config + model + pipeline + position |
| `executor.py` | Translates signals into MT5 order requests |
| `trade_logger.py` | CSV trade log writer |
| `reporter.py` | Performance metrics from trade log |
| `health.py` | Heartbeat, reconnect, weekend detection |
| `main.py` | CLI entry point (`trade2-live`) |

### Key Design Decisions

- **Strategy isolation via magic numbers**: Each strategy gets a unique MT5 magic number so positions don't conflict
- **Server-side SL/TP**: MT5 handles stop execution even if the script is offline
- **Trailing stop updates**: Recomputed on each 5M bar, modified via MT5 API
- **Crash recovery**: On startup, discover orphaned positions and resume management
- **Backtest-matched sizing**: `equity * base_alloc * position_size_mult / price / 100` = lots

### Reused Existing Code (no modifications)

- `trade2/config/loader.py` — `load_config()`
- `trade2/features/builder.py` — `add_1h_features()`, `add_5m_features()`
- `trade2/features/hmm_features.py` — `get_hmm_feature_matrix()`
- `trade2/models/hmm.py` — `XAUUSDRegimeModel.load()`, `.regime_labels()`, `.bull_probability()`, `.bear_probability()`
- `trade2/signals/regime.py` — `forward_fill_1h_regime()`
- `trade2/signals/router.py` — `route_signals()`
- `trade2/signals/generator.py` — `compute_stops_regime_aware()`

### Trade Logging

CSV per strategy at `artefacts/live_trades/live_trades_{name}.csv` with fields: timestamp, ticket, strategy, direction, entry/exit price, SL, TP, lots, signal_source, regime, bull_prob, PnL, exit_reason, duration.

### Performance Reporting

Every 4 hours (or on demand via `trade2-live --report`): annualized return, Sharpe, drawdown, win rate, profit factor. Compared to backtest metrics to flag divergence. Output to `artefacts/live_reports/`.

### Error Handling

- MT5 disconnect: exponential backoff reconnect (5s -> 300s)
- Order failure: retry 3x with 2s delay
- Crash recovery: discover positions on startup
- Weekend: skip processing
- Stale data: warn >2min, pause >10min

## Out of Scope

- Multi-symbol support (XAUUSD only)
- Web dashboard for live monitoring
- SMS/email alerting
- Auto-switching between demo and live based on performance thresholds
