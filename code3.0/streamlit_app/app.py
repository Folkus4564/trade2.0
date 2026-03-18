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
from components.charts         import candlestick_chart, equity_curve_chart, monthly_pnl_chart, trade_analysis_chart
from components.metrics_table  import render_metric_cards, render_full_metrics_table
from components.scheme_search  import render_scheme_search
from components.tv_research    import render_tv_research
from components.trade_log           import render_trade_log
from components.approved_strategies import render_approved_strategies
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

# ---- Tabs ----
tab_backtest, tab_tradelog, tab_approved, tab_scheme, tab_tv = st.tabs(["Strategy Backtest", "Trade Log", "Approved Strategies", "Scheme Search", "TV Indicator Research"])

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
artefacts_dir = str(_REPO_ROOT / "code3.0" / "artefacts")

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

# ---- Session state ----
if "saved_run" not in st.session_state:
    st.session_state["saved_run"] = None
if "current_run" not in st.session_state:
    st.session_state["current_run"] = None

# ---- Run backtest ----
if config.get("_ui_run") and hmm is not None:
    from trade2.backtesting.costs import doubled_costs
    backtest_config = doubled_costs(config) if config.get("_ui_2x_cost") else config
    with st.spinner("Building features and running backtest..."):
        feat_1h, feat_5m = build_features(df_1h, df_5m, config)
        run_result = generate_and_backtest(feat_1h, feat_5m, hmm, backtest_config, split)
        st.session_state["current_run"] = run_result

if config.get("_ui_save") and st.session_state["current_run"] is not None:
    st.session_state["saved_run"] = st.session_state["current_run"]
    st.sidebar.success("Run saved for comparison!")

current = st.session_state["current_run"]
saved   = st.session_state["saved_run"]

# ========================================================
# TAB 1 — Strategy Backtest
# ========================================================
with tab_backtest:
    if hmm is None:
        st.warning("HMM model not found at artefacts/models/hmm_regime_model.pkl. Run trade2 --retrain-model first.")

    if current is not None:
        metrics      = current["metrics"]
        signals      = current["signals"]
        equity_curve = current.get("equity_curve")

        cmp_metrics = saved["metrics"] if saved is not None else None
        trades = current.get("trades")

        # Row 1: Metric cards
        cost_label = " — 2x Cost Sensitivity" if config.get("_ui_2x_cost") else ""
        st.subheader(f"Performance ({split.upper()}){cost_label}")
        render_metric_cards(metrics, comparison=cmp_metrics)

        # Row 2: Charts
        col1, col2 = st.columns([3, 2])

        with col1:
            st.plotly_chart(
                candlestick_chart(df_5m, signals, trades=trades if trades is not None and len(trades) > 0 else None, tail_bars=500),
                use_container_width=True,
            )

        with col2:
            if equity_curve is not None:
                st.plotly_chart(equity_curve_chart(equity_curve), use_container_width=True)
            else:
                st.info("Equity data not available. Check run_backtest output format.")

        # Row 3: Strategy breakdown
        if signals is not None and "signal_source" in signals.columns:
            with st.expander("Strategy Breakdown", expanded=False):
                import pandas as pd
                all_sigs = pd.concat([
                    signals[signals["signal_long"] == 1].assign(_dir="long"),
                    signals[signals["signal_short"] == 1].assign(_dir="short"),
                ])
                if len(all_sigs) > 0:
                    breakdown = (
                        all_sigs.groupby(["signal_source", "_dir"])
                        .size()
                        .unstack(fill_value=0)
                        .rename_axis("Strategy")
                    )
                    if "cdc_zone_green" in signals.columns:
                        zone_counts = {
                            "green":  int(signals["cdc_zone_green"].sum()),
                            "yellow": int(signals["cdc_zone_yellow"].sum()),
                            "blue":   int(signals["cdc_zone_blue"].sum()),
                            "red":    int(signals["cdc_zone_red"].sum()),
                        }
                        st.caption("CDC Zone Distribution (bars)")
                        zcols = st.columns(4)
                        for col_w, (zone, cnt) in zip(zcols, zone_counts.items()):
                            col_w.metric(zone.capitalize(), cnt)
                    st.dataframe(breakdown, use_container_width=True)

        # Row 4: Trade visualisation + log
        if trades is not None and len(trades) > 0:
            import pandas as pd

            # Monthly PnL
            st.plotly_chart(monthly_pnl_chart(trades), use_container_width=True)

            # PnL waterfall + duration + exit reasons
            st.plotly_chart(trade_analysis_chart(trades), use_container_width=True)

            # Trade log table
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
                if "size" in log.columns:
                    log["size"] = log["size"].round(4)
                if "duration_bars" in log.columns:
                    log["hold_h"] = (log["duration_bars"] * 5 / 60).round(1)
                display_cols = [c for c in [
                    "entry_time", "exit_time", "direction", "lots",
                    "entry_price", "sl", "tp", "exit_price",
                    "hold_h", "pnl", "exit_reason",
                ] if c in log.columns]
                styled = log[display_cols].style.map(
                    lambda v: "color: #00e676" if isinstance(v, (int, float)) and v > 0 else
                              "color: #ff5252" if isinstance(v, (int, float)) and v < 0 else "",
                    subset=["pnl"] if "pnl" in display_cols else []
                )
                st.dataframe(styled, use_container_width=True, height=400)

        # Row 5: Full metrics table + comparison
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
        st.plotly_chart(
            candlestick_chart(df_5m.tail(500), None),
            use_container_width=True,
        )

# ========================================================
# TAB 2 — Trade Log
# ========================================================
with tab_tradelog:
    live_trades = current.get("trades") if current is not None else None
    render_trade_log(artefacts_dir, live_trades=live_trades)

# ========================================================
# TAB 3 — Approved Strategies
# ========================================================
with tab_approved:
    render_approved_strategies(artefacts_dir)

# ========================================================
# TAB 4 — Scheme Search
# ========================================================
with tab_scheme:
    render_scheme_search(artefacts_dir)

# ========================================================
# TAB 5 — TV Indicator Research
# ========================================================
with tab_tv:
    render_tv_research(artefacts_dir)
