# Golden Model Dashboard — Design Spec
Date: 2026-03-17

## Goal

Add a dedicated "Golden Model" page to the existing Streamlit dashboard. The page auto-loads the best golden model on startup, lets the user pick from all saved golden models via a dropdown, and provides a full interactive backtest explorer (same charts and metrics as the main Strategy Backtest tab).

## Approach

Streamlit multi-page app: new file `code3.0/streamlit_app/pages/1_Golden_Model.py`. Streamlit (v1.45.1) auto-discovers it and adds it to the sidebar nav. No changes to `app.py`.

## Files

### New
- `code3.0/streamlit_app/pages/` — directory must be created (Streamlit multi-page requirement)
- `code3.0/streamlit_app/pages/1_Golden_Model.py` — the page itself

### Modified
- `code3.0/streamlit_app/utils/pipeline_runner.py` — add two helpers:
  1. `load_hmm_model_from_path(path: str)` — loads a `.pkl` via `XAUUSDRegimeModel.load(path)` (from `trade2.models.hmm`), decorated with `@st.cache_resource` keyed on `path`
  2. `load_base_config(repo_root: str) -> dict` — calls `load_config(str(Path(repo_root) / "code3.0/configs/base.yaml"))` and returns the config dict, decorated with `@st.cache_data`
- `code3.0/streamlit_app/app.py` — replace private `_load_base_config()` with a call to the new shared `load_base_config(repo_root)` to avoid duplication

## Page File Requirements

`pages/1_Golden_Model.py` must open with:
1. `sys.path` preamble — same as `app.py` lines 10-13: insert `code3.0/src` onto `sys.path` before any `trade2` import
2. `st.set_page_config(...)` — must be the very first Streamlit call (Streamlit requirement for multi-page apps)

## UI Layout

### Top bar
- **Golden model dropdown** — lists all `*.pkl` in `artefacts/models/golden/`, sorted by `annualized_return` descending (from sidecar). Label format: `ret56% sh2.10 — hmm_4h_2states_2026_03_17`. Note: `annualized_return` in the sidecar is a decimal fraction (e.g., `0.5636`); multiply by 100 **only for the dropdown label string** (e.g., `f"ret{v*100:.0f}%"`). Models whose sidecar is missing sort last and show "N/A" in the banner.
- **Static metrics banner** — loaded from the `_metrics.json` sidecar (no backtest). Shows: Return, Sharpe, DD, Win Rate, PF, Trades as metric cards using `render_metric_cards`. Pass raw decimal values directly to `render_metric_cards` — it applies `:.1%` formatting internally, so do **not** multiply by 100 before passing. Updates automatically on dropdown change (standard Streamlit re-run). If sidecar is missing, show an info message instead.

### Sidebar
- Split selector (train / val / test), defaulting to `test`
- Strategy config controls via `render_sidebar(base_config)` — the page manages its own `_ui_run` flag via a local "Run Backtest" button; it does **not** rely on `render_sidebar`'s auto-run logic
- `_ui_run` is set to `True` only on explicit button click using a page-local condition, not the session-state-based auto-run in the main app
- Config is loaded via `load_base_config(repo_root)` (new helper in `pipeline_runner.py`)

### Session State
The golden model page uses distinct session state keys to avoid collision with `app.py`:
- `st.session_state["golden_current_run"]`
- `st.session_state["golden_saved_run"]`

### Pre-run placeholder
Before any backtest is run, display the static metrics banner (from sidecar) and a candlestick chart of the last 500 bars of `df_5m` with no signals, analogous to the main app's fallback view.

### Main area (post-run)
- Metric cards (live backtest results via `render_metric_cards`)
- Candlestick chart + equity curve (2-column); candlestick uses `df_5m` for the selected split
- Monthly PnL chart
- Trade analysis chart
- Full trade log (expandable)

## Data Flow

1. **Page load** — scan `artefacts/models/golden/` for `*.pkl`. For each, derive sidecar path as `pkl_path.with_name(pkl_path.stem + "_metrics.json")`. Read sidecar if present. Sort by `annualized_return` desc (missing sidecars sort last). Populate dropdown. Render static banner from top model's sidecar.
2. **Dropdown change** — reload sidecar → re-render banner. No backtest triggered (standard Streamlit re-run).
3. **"Run Backtest" click** — load selected `.pkl` via `load_hmm_model_from_path(path)` (cached by path), load data (cached), build features, call `generate_and_backtest(feat_1h, feat_5m, hmm, config, split)` with the full base config, render full results into `st.session_state["golden_current_run"]`.
4. **Config** — `load_base_config(repo_root)` loads full `base.yaml`-derived config (with all required keys including `backtest.init_cash`). The golden model path is not injected into config — instead, the HMM object is loaded separately and passed directly to `generate_and_backtest`.

## Sidecar Convention
- Path: `{pkl_stem}_metrics.json` (e.g., `hmm_4h_2states_2026_03_17_ret56pct_sh2.10_metrics.json`)
- Structure: `{"test_metrics": {"annualized_return": 0.5636, "sharpe_ratio": 2.0954, "max_drawdown": -0.1282, "win_rate": 0.5902, "profit_factor": 2.0326, "total_trades": 61, ...}}`
- `annualized_return` is a decimal fraction — multiply by 100 for % display

## Reused Components (no changes)
- `components/charts.py`
- `components/metrics_table.py`
- `components/sidebar.py` (render_sidebar used for config, _ui_run managed separately)

## Out of Scope
- Comparison view between two golden models
- Auto-refresh / live monitoring
- Model deletion or management UI
