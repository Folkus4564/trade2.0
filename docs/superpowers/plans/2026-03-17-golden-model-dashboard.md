# Golden Model Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit multi-page "Golden Model" page that auto-loads the best golden model, shows a metrics banner from the sidecar, and lets the user run a full interactive backtest.

**Architecture:** New `pages/1_Golden_Model.py` reuses all existing chart and metrics components. Two helper functions (`load_hmm_model_from_path`, `load_base_config`) are added to `pipeline_runner.py`; `app.py` is updated to use the shared config loader. The page manages its own session state keys to avoid collision with the main app.

**Tech Stack:** Python 3.10+, Streamlit 1.45.1, Plotly, trade2 package (installed via `pip install -e code3.0/`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `code3.0/streamlit_app/utils/pipeline_runner.py` | Modify | Add `load_hmm_model_from_path(path)` and `load_base_config(repo_root)` |
| `code3.0/streamlit_app/app.py` | Modify | Replace private `_load_base_config()` with call to shared helper |
| `code3.0/streamlit_app/pages/` | Create dir | Streamlit multi-page discovery requires this directory |
| `code3.0/streamlit_app/pages/1_Golden_Model.py` | Create | Golden model page: dropdown, sidecar banner, backtest explorer |
| `code3.0/tests/test_pipeline_runner_helpers.py` | Create | Unit tests for the two new helpers |

---

## Task 1: Add shared helpers to pipeline_runner.py

**Files:**
- Modify: `code3.0/streamlit_app/utils/pipeline_runner.py`
- Create: `code3.0/tests/test_pipeline_runner_helpers.py`

### Context

`pipeline_runner.py` currently has a private `_ROOT` constant (line 14) pointing to `code3.0/` (two parents up from `utils/`). The file already imports `XAUUSDRegimeModel` indirectly in `load_hmm_model`. We need two new exported functions:

1. `load_hmm_model_from_path(path: str)` — loads any `.pkl` by absolute path using `XAUUSDRegimeModel.load(path)`, cached with `@st.cache_resource` keyed on `path`.
2. `load_base_config(repo_root: str) -> dict` — loads `{repo_root}/code3.0/configs/base.yaml` using `trade2.config.loader.load_config`, cached with `@st.cache_data`.

- [ ] **Step 1: Write the failing tests**

Create `code3.0/tests/test_pipeline_runner_helpers.py`:

```python
"""Tests for new pipeline_runner helpers (no Streamlit context needed)."""
import sys
from pathlib import Path

# Put trade2 on path
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "code3.0" / "src"))

import pytest


def test_load_base_config_returns_dict():
    """load_base_config loads base.yaml and returns a dict with expected top-level keys."""
    import importlib, types
    # Stub streamlit so pipeline_runner imports without a running Streamlit server
    st_stub = types.ModuleType("streamlit")
    st_stub.cache_data     = lambda **kw: (lambda f: f)
    st_stub.cache_resource = lambda **kw: (lambda f: f)
    sys.modules["streamlit"] = st_stub

    # Now import the function under test
    sys.path.insert(0, str(_ROOT / "code3.0" / "streamlit_app"))
    from utils.pipeline_runner import load_base_config  # noqa: E402

    repo_root = str(_ROOT)
    cfg = load_base_config(repo_root)

    assert isinstance(cfg, dict)
    for key in ("data", "hmm", "risk", "backtest", "strategies"):
        assert key in cfg, f"Expected top-level key '{key}' in config"


def test_load_hmm_model_from_path_loads_golden(tmp_path):
    """load_hmm_model_from_path loads a .pkl that XAUUSDRegimeModel can read."""
    import importlib, types, pickle
    st_stub = types.ModuleType("streamlit")
    st_stub.cache_data     = lambda **kw: (lambda f: f)
    st_stub.cache_resource = lambda **kw: (lambda f: f)
    sys.modules["streamlit"] = st_stub

    sys.path.insert(0, str(_ROOT / "code3.0" / "streamlit_app"))
    from utils.pipeline_runner import load_hmm_model_from_path  # noqa: E402

    # Use the real best golden model if it exists, else skip
    golden_dir = _ROOT / "code3.0" / "artefacts" / "models" / "golden"
    pkls = sorted(golden_dir.glob("*.pkl"))
    if not pkls:
        pytest.skip("No golden models found")

    model = load_hmm_model_from_path(str(pkls[-1]))
    assert model is not None
    assert hasattr(model, "predict"), "Model must have a .predict() method"
    assert hasattr(model, "state_map"), "Model must have a .state_map dict"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
python -m pytest code3.0/tests/test_pipeline_runner_helpers.py -v 2>&1 | head -40
```

Expected: `ImportError` or `AttributeError` — `load_base_config` and `load_hmm_model_from_path` don't exist yet.

- [ ] **Step 3: Add helpers to pipeline_runner.py**

Open `code3.0/streamlit_app/utils/pipeline_runner.py`. After the existing `load_hmm_model` function (line 38), add:

```python
@st.cache_resource(show_spinner="Loading model...")
def load_hmm_model_from_path(path: str):
    """Load any golden model .pkl by absolute path. Cached by path string."""
    from trade2.models.hmm import XAUUSDRegimeModel
    return XAUUSDRegimeModel.load(path)


@st.cache_data(show_spinner="Loading config...")
def load_base_config(repo_root: str):
    """Load base.yaml from repo_root/code3.0/configs/base.yaml. Cached."""
    from trade2.config.loader import load_config
    cfg_path = str(Path(repo_root) / "code3.0" / "configs" / "base.yaml")
    return load_config(cfg_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
python -m pytest code3.0/tests/test_pipeline_runner_helpers.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Update app.py to use load_base_config**

In `code3.0/streamlit_app/app.py`:

Replace lines 40-46 (the private `_load_base_config` function and its call):
```python
@st.cache_data
def _load_base_config():
    from trade2.config.loader import load_config
    cfg_path = _REPO_ROOT / "code3.0" / "configs" / "base.yaml"
    return load_config(str(cfg_path))

base_config = _load_base_config()
```

With:
```python
from utils.pipeline_runner import load_base_config
base_config = load_base_config(str(_REPO_ROOT))
```

- [ ] **Step 6: Smoke-test the existing app still loads**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
python -c "
import sys; sys.path.insert(0, 'code3.0/streamlit_app'); sys.path.insert(0, 'code3.0/src')
from utils.pipeline_runner import load_base_config
cfg = load_base_config('C:/Users/LENOVO/Desktop/trade2.0')
print('OK:', list(cfg.keys())[:5])
"
```

Expected: `OK: ['data', 'hmm', ...]` (no errors).

- [ ] **Step 7: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/streamlit_app/utils/pipeline_runner.py code3.0/streamlit_app/app.py code3.0/tests/test_pipeline_runner_helpers.py
git commit -m "feat: add load_hmm_model_from_path and load_base_config helpers"
```

---

## Task 2: Create the Golden Model page

**Files:**
- Create: `code3.0/streamlit_app/pages/` (directory)
- Create: `code3.0/streamlit_app/pages/1_Golden_Model.py`

### Context

- `render_sidebar` (sidebar.py line 120) sets `cfg["_ui_run"] = run_clicked or (st.session_state.get("current_run") is None)`. This auto-fires on the first load and checks `"current_run"` — which belongs to the main app. On this page we control `_ui_run` ourselves and use session keys `"golden_current_run"` / `"golden_saved_run"`.
- `render_metric_cards` accepts raw decimal fractions (`annualized_return=0.5636`) and formats them as `:.1%` internally — do NOT multiply by 100 before passing.
- The dropdown label uses a manual f-string, so multiply by 100 there: `f"ret{v*100:.0f}%"`.
- The sidecar path is derived as: `pkl_path.with_name(pkl_path.stem + "_metrics.json")`.

- [ ] **Step 1: Create the pages directory**

```bash
mkdir -p "C:/Users/LENOVO/Desktop/trade2.0/code3.0/streamlit_app/pages"
```

- [ ] **Step 2: Write 1_Golden_Model.py**

Create `code3.0/streamlit_app/pages/1_Golden_Model.py`:

```python
"""
Golden Model Dashboard — pages/1_Golden_Model.py

Streamlit multi-page app page. Auto-loads the best golden model,
shows a static metrics banner from the sidecar, and runs a full
interactive backtest on demand.

Run (from repo root):
    streamlit run code3.0/streamlit_app/app.py
"""
import sys
from pathlib import Path

# -- sys.path preamble (same as app.py) --
_REPO_ROOT = Path(__file__).parents[3]
_PKG_SRC   = _REPO_ROOT / "code3.0" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))
# Also ensure streamlit_app/ is on path for component imports
_APP_DIR = Path(__file__).parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import json
import hashlib
import streamlit as st

# -- Must be first Streamlit call (Streamlit multi-page requirement) --
st.set_page_config(
    page_title="Golden Model — XAUUSD",
    page_icon=":trophy:",
    layout="wide",
)

from components.charts        import candlestick_chart, equity_curve_chart, monthly_pnl_chart, trade_analysis_chart
from components.metrics_table import render_metric_cards, render_full_metrics_table
from components.sidebar       import render_sidebar
from utils.pipeline_runner    import (
    load_data,
    load_hmm_model_from_path,
    load_base_config,
    build_features,
    generate_and_backtest,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
base_config = load_base_config(str(_REPO_ROOT))
artefacts_dir = _REPO_ROOT / "code3.0" / "artefacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_golden_models(golden_dir: Path) -> list:
    """
    Return list of dicts sorted by annualized_return descending.
    Each dict: {path, stem, metrics (or None), label}
    Models without a sidecar sort last.
    """
    results = []
    for pkl in golden_dir.glob("*.pkl"):
        sidecar = pkl.with_name(pkl.stem + "_metrics.json")
        metrics = None
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    data = json.load(f)
                metrics = data.get("test_metrics", {})
            except Exception:
                pass
        ret = metrics.get("annualized_return", -999) if metrics else -999
        sh  = metrics.get("sharpe_ratio", 0.0)       if metrics else 0.0
        if metrics:
            label = f"ret{ret*100:.0f}% sh{sh:.2f} — {pkl.stem}"
        else:
            label = f"N/A — {pkl.stem}"
        results.append({"path": pkl, "stem": pkl.stem, "metrics": metrics, "label": label, "_sort": ret})

    results.sort(key=lambda x: x["_sort"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "golden_current_run" not in st.session_state:
    st.session_state["golden_current_run"] = None
if "golden_saved_run" not in st.session_state:
    st.session_state["golden_saved_run"] = None

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
st.title("Golden Model Explorer")
st.caption("Load any saved golden model, inspect its sidecar metrics, and run a live backtest.")

# ---------------------------------------------------------------------------
# Golden model selector
# ---------------------------------------------------------------------------
golden_dir = artefacts_dir / "models" / "golden"
models = _scan_golden_models(golden_dir)

if not models:
    st.error(f"No golden models found in {golden_dir}. Run trade2 to produce a golden model first.")
    st.stop()

labels  = [m["label"] for m in models]
chosen_idx = st.selectbox(
    "Select golden model",
    range(len(labels)),
    format_func=lambda i: labels[i],
    index=0,
)
chosen = models[chosen_idx]

# Static sidecar metrics banner
st.subheader("Saved Metrics (from sidecar)")
if chosen["metrics"]:
    render_metric_cards(chosen["metrics"])
else:
    st.info("No sidecar metrics available for this model.")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar (config + run button)
# ---------------------------------------------------------------------------
config = render_sidebar(base_config)
split  = config.get("_ui_split", "test")

# Override auto-run logic: on this page we only run on explicit button click.
# render_sidebar sets _ui_run = run_clicked OR (current_run is None) where
# "current_run" belongs to the main app's session state — not this page.
# We ignore render_sidebar's _ui_run entirely and use a dedicated page-local button.
run_triggered = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
config_hash = hashlib.md5(
    json.dumps(
        {k: v for k, v in config.items() if not k.startswith("_ui")},
        sort_keys=True, default=str,
    ).encode()
).hexdigest()

data_splits = load_data(config_hash, config)
df_1h = data_splits["1H"][split]
df_5m = data_splits["5M"][split]

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------
if run_triggered:
    with st.spinner("Loading model and running backtest..."):
        hmm = load_hmm_model_from_path(str(chosen["path"]))
        feat_1h, feat_5m = build_features(df_1h, df_5m, config)
        run_result = generate_and_backtest(feat_1h, feat_5m, hmm, config, split)
        st.session_state["golden_current_run"] = run_result

if config.get("_ui_save") and st.session_state["golden_current_run"] is not None:
    st.session_state["golden_saved_run"] = st.session_state["golden_current_run"]
    st.sidebar.success("Run saved for comparison!")

current = st.session_state["golden_current_run"]
saved   = st.session_state["golden_saved_run"]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if current is not None:
    metrics      = current["metrics"]
    signals      = current["signals"]
    equity_curve = current.get("equity_curve")
    trades       = current.get("trades")
    cmp_metrics  = saved["metrics"] if saved is not None else None

    st.subheader(f"Backtest Results ({split.upper()})")
    render_metric_cards(metrics, comparison=cmp_metrics)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(
            candlestick_chart(
                df_5m, signals,
                trades=trades if trades is not None and len(trades) > 0 else None,
                tail_bars=500,
            ),
            use_container_width=True,
        )
    with col2:
        if equity_curve is not None:
            st.plotly_chart(equity_curve_chart(equity_curve), use_container_width=True)
        else:
            st.info("No equity curve data.")

    if trades is not None and len(trades) > 0:
        import pandas as pd
        st.plotly_chart(monthly_pnl_chart(trades), use_container_width=True)
        st.plotly_chart(trade_analysis_chart(trades), use_container_width=True)

        with st.expander(f"Full Trade Log ({len(trades)} trades)", expanded=False):
            log = trades.copy()
            for col in ["entry_time", "exit_time"]:
                if col in log.columns:
                    log[col] = pd.to_datetime(log[col]).dt.strftime("%Y-%m-%d %H:%M")
            for col in ["entry_price", "exit_price", "sl", "tp"]:
                if col in log.columns:
                    log[col] = log[col].round(2)
            if "pnl" in log.columns:
                log["pnl"] = log["pnl"].round(2)
            if "duration_bars" in log.columns:
                log["hold_h"] = (log["duration_bars"] * 5 / 60).round(1)
            display_cols = [c for c in [
                "entry_time", "exit_time", "direction",
                "entry_price", "sl", "tp", "exit_price",
                "hold_h", "pnl", "exit_reason",
            ] if c in log.columns]
            styled = log[display_cols].style.map(
                lambda v: "color: #00e676" if isinstance(v, (int, float)) and v > 0 else
                          "color: #ff5252" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["pnl"] if "pnl" in display_cols else [],
            )
            st.dataframe(styled, use_container_width=True, height=400)

    with st.expander("Full Metrics", expanded=False):
        if saved is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Current Run")
                render_full_metrics_table(metrics)
            with c2:
                st.caption("Saved Run")
                render_full_metrics_table(saved["metrics"])
        else:
            render_full_metrics_table(metrics)

else:
    # Pre-run placeholder
    st.info("Select a model above, configure the sidebar, then click **Run Backtest**.")
    st.plotly_chart(
        candlestick_chart(df_5m.tail(500), None),
        use_container_width=True,
    )
```

- [ ] **Step 3: Verify the page imports without errors**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
python -c "
import sys
sys.path.insert(0, 'code3.0/streamlit_app')
sys.path.insert(0, 'code3.0/src')

# Stub streamlit
import types
st = types.ModuleType('streamlit')
st.set_page_config = lambda **kw: None
st.cache_data      = lambda **kw: (lambda f: f)
st.cache_resource  = lambda **kw: (lambda f: f)
sys.modules['streamlit'] = st

from utils.pipeline_runner import load_base_config, load_hmm_model_from_path
cfg = load_base_config('C:/Users/LENOVO/Desktop/trade2.0')
print('Config keys:', list(cfg.keys())[:5])

from pathlib import Path
import json
golden_dir = Path('code3.0/artefacts/models/golden')
pkls = sorted(golden_dir.glob('*.pkl'))
print('Golden models found:', len(pkls))
for p in pkls:
    sidecar = p.with_name(p.stem + '_metrics.json')
    print(' ', p.name, '| sidecar exists:', sidecar.exists())
"
```

Expected: config keys printed, golden models listed with sidecars found.

- [ ] **Step 4: Run the full Streamlit app and navigate to Golden Model page**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
streamlit run code3.0/streamlit_app/app.py
```

Open browser. Verify:
- Sidebar nav shows "1 Golden Model" page
- Navigating to it shows the dropdown defaulting to `ret56% sh2.10 — hmm_4h_2states_2026_03_17`
- Sidecar metrics banner shows: Return ~56%, Sharpe ~2.10, DD ~-12.8%, Trades 61
- Clicking "Run Backtest" runs the backtest and shows charts + trade log

- [ ] **Step 5: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/streamlit_app/pages/
git commit -m "feat: add Golden Model Explorer Streamlit page"
```

---

## Task 3: Push to GitHub

- [ ] **Step 1: Push**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git push
```

Expected: branch pushed, no errors.
