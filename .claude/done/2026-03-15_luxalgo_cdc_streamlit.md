# Plan: LuxAlgo SMC + CDC Action Zone + Streamlit Dashboard

**Date planned:** 2026-03-15
**Motivation:** Add LuxAlgo-style SMC structural features, a CDC trend-zone sub-strategy, and an interactive Streamlit dashboard for live strategy exploration.

## Files Created
- `code3.0/src/trade2/features/cdc.py` — CDC Action Zone (OHLC4 EMA zones, buy/sell transitions)
- `code3.0/src/trade2/features/smc_luxalgo.py` — LuxAlgo fractal swings, BOS/CHoCH, premium/discount, equal H/L
- `code3.0/src/trade2/signals/strategies/cdc.py` — CDC sub-strategy (ADX, BOS confirm, regime gating, sizing)
- `code3.0/streamlit_app/app.py` — Main dashboard entry point
- `code3.0/streamlit_app/components/sidebar.py` — Strategy toggles + param sliders
- `code3.0/streamlit_app/components/charts.py` — Plotly candlestick + equity curve
- `code3.0/streamlit_app/components/metrics_table.py` — Metric cards + comparison mode
- `code3.0/streamlit_app/utils/pipeline_runner.py` — Cached pipeline wrappers

## Files Modified
- `code3.0/configs/base.yaml` — Added `strategies.cdc`, `smc_luxalgo`, `smc_luxalgo_5m` sections
- `code3.0/src/trade2/features/builder.py` — Conditional CDC + LuxAlgo calls in add_1h/5m_features()
- `code3.0/src/trade2/signals/router.py` — `_empty_signals()`, enabled guards, CDC as 4th strategy slot
- `code3.0/src/trade2/signals/generator.py` — "cdc" added to compute_stops_regime_aware() loop
- `code3.0/pyproject.toml` — `[project.optional-dependencies] ui = [streamlit, plotly]`

## Phases
1. CDC Action Zone — DONE
2. LuxAlgo SMC features — DONE
3. Streamlit UI — DONE
4. BOS/CHoCH confirm on trend.py + cdc.py — DONE

## Completed
**Date:** 2026-03-15
**Commits:** d4cabff (initial), 904d704 (Phase 4 + improvement roadmap)
