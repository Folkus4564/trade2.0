# Golden Model Dashboard — Design Spec
Date: 2026-03-17

## Goal

Add a dedicated "Golden Model" page to the existing Streamlit dashboard. The page auto-loads the best golden model on startup, lets the user pick from all saved golden models via a dropdown, and provides a full interactive backtest explorer (same charts and metrics as the main Strategy Backtest tab).

## Approach

Streamlit multi-page app: new file `code3.0/streamlit_app/pages/1_Golden_Model.py`. Streamlit auto-discovers it and adds it to the sidebar nav. No changes to `app.py`.

## Files

### New
- `code3.0/streamlit_app/pages/1_Golden_Model.py` — the page itself

### Modified
- `code3.0/streamlit_app/utils/pipeline_runner.py` — add `load_hmm_model_from_path(path: str)` that loads a `.pkl` directly by absolute path (3-line addition)

## UI Layout

### Top bar
- **Golden model dropdown** — lists all `*.pkl` in `artefacts/models/golden/`, sorted by `annualized_return` descending. Label format: `ret56% sh2.10 — hmm_4h_2states_2026_03_17`. Defaults to the highest-return model.
- **Static metrics banner** — loaded instantly from the `_metrics.json` sidecar (no backtest). Shows: Return, Sharpe, DD, Win Rate, PF, Trades as metric cards. Updates on dropdown change.

### Sidebar
- Split selector (train / val / test), defaulting to `test`
- Strategy config controls via existing `render_sidebar` logic (model-selection parts skipped — model comes from dropdown)
- "Run Backtest" button

### Main area (post-run)
- Metric cards (live backtest results)
- Candlestick chart + equity curve (2-column)
- Monthly PnL chart
- Trade analysis chart
- Full trade log (expandable)

## Data Flow

1. **Page load** — scan `artefacts/models/golden/` for `*.pkl`, read `_metrics.json` sidecars, sort by return desc, populate dropdown, render static banner from top model's sidecar.
2. **Dropdown change** — reload sidecar → re-render banner. No backtest triggered.
3. **Run Backtest click** — load selected `.pkl` via `load_hmm_model_from_path()`, load data (cached), build features, call `generate_and_backtest()`, render full results.
4. **Config** — `render_sidebar` produces config dict; golden model path injected before calling `generate_and_backtest`.

## Reused Components (no changes)
- `components/charts.py`
- `components/metrics_table.py`
- `utils/pipeline_runner.py` (except the one addition)

## Out of Scope
- Comparison view between two golden models
- Auto-refresh / live monitoring
- Model deletion or management UI
