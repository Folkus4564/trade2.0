"""
XAUUSD Strategy Dashboard
Run: streamlit run code3.0/streamlit_app/app.py
"""
import sys
import os
from pathlib import Path

# Ensure trade2 on path
_REPO_ROOT = Path(__file__).parents[2]
_PKG_SRC   = _REPO_ROOT / "code3.0" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import streamlit as st

from components.sidebar        import render_sidebar
from components.charts         import candlestick_chart, equity_curve_chart
from components.metrics_table  import render_metric_cards, render_full_metrics_table
from utils.pipeline_runner     import (
    load_data, load_hmm_model, build_features,
    get_regime, generate_and_backtest,
)


# ---- Page config ----
st.set_page_config(
    page_title="XAUUSD Strategy Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("XAUUSD Strategy Dashboard")

# ---- Load base config ----
@st.cache_data
def _load_base_config():
    from trade2.config.loader import load_config
    cfg_path = _REPO_ROOT / "code3.0" / "configs" / "base.yaml"
    return load_config(str(cfg_path))

base_config = _load_base_config()

# ---- Sidebar ----
config = render_sidebar(base_config)
split  = config.get("_ui_split", "test")

# ---- Artefacts dir ----
artefacts_dir = str(_REPO_ROOT / "artefacts")

# ---- Load data (cached) ----
import hashlib, json
config_hash = hashlib.md5(json.dumps({
    k: v for k, v in config.items() if not k.startswith("_ui")
}, sort_keys=True, default=str).encode()).hexdigest()

data_splits = load_data(config_hash, config)
df_1h_all   = data_splits["1H"]
df_5m_all   = data_splits["5M"]

df_1h = df_1h_all[split]
df_5m = df_5m_all[split]

# ---- Load HMM model ----
hmm = load_hmm_model(config, artefacts_dir)
if hmm is None:
    st.warning("HMM model not found at artefacts/hmm_regime_model.pkl. Run trade2 --retrain-model first.")

# ---- Session state ----
if "saved_run" not in st.session_state:
    st.session_state["saved_run"] = None
if "current_run" not in st.session_state:
    st.session_state["current_run"] = None

# ---- Run backtest ----
if config.get("_ui_run") and hmm is not None:
    with st.spinner("Building features and running backtest..."):
        feat_1h, feat_5m = build_features(df_1h, df_5m, config)
        run_result = generate_and_backtest(feat_1h, feat_5m, hmm, config, split)
        st.session_state["current_run"] = run_result

if config.get("_ui_save") and st.session_state["current_run"] is not None:
    st.session_state["saved_run"] = st.session_state["current_run"]
    st.sidebar.success("Run saved for comparison!")

current = st.session_state["current_run"]
saved   = st.session_state["saved_run"]

# ---- Layout ----
if current is not None:
    metrics    = current["metrics"]
    signals    = current["signals"]
    results    = current.get("results")

    cmp_metrics = saved["metrics"] if saved is not None else None

    # Row 1: Metric cards
    st.subheader(f"Performance ({split.upper()})")
    render_metric_cards(metrics, comparison=cmp_metrics)

    # Row 2: Charts
    col1, col2 = st.columns([3, 2])

    with col1:
        st.plotly_chart(
            candlestick_chart(df_5m, signals, tail_bars=500),
            use_container_width=True,
        )

    with col2:
        if results is not None and hasattr(results, "columns") and "equity" in results.columns:
            st.plotly_chart(equity_curve_chart(results), use_container_width=True)
        else:
            st.info("Equity data not available. Check run_backtest output format.")

    # Row 3: Full metrics table + comparison
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
    st.info("Configure strategy in sidebar, then click **Run Backtest** to see results.")

    # Show empty chart placeholder
    st.plotly_chart(
        candlestick_chart(df_5m.tail(500), None),
        use_container_width=True,
    )
