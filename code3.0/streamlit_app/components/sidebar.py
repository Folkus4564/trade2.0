"""
components/sidebar.py - Strategy parameter sidebar for Streamlit.
"""
import copy
from typing import Dict, Any
import streamlit as st


def render_sidebar(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render sidebar controls and return a deep-copied config
    with user-modified values.

    Args:
        config: Base config dict loaded from base.yaml.

    Returns:
        Modified config dict.
    """
    cfg = copy.deepcopy(config)

    st.sidebar.title("Strategy Controls")

    # ---- Data split selector ----
    st.sidebar.header("Data Split")
    split = st.sidebar.radio("Split", ["train", "val", "test"], index=2)
    cfg["_ui_split"] = split

    # ---- Strategy toggles ----
    st.sidebar.header("Strategy Toggles")
    cfg["strategies"]["trend"]["enabled"]    = st.sidebar.checkbox("Trend Strategy",    value=cfg["strategies"]["trend"].get("enabled", True))
    cfg["strategies"]["range"]["enabled"]    = st.sidebar.checkbox("Range Strategy",    value=cfg["strategies"]["range"].get("enabled", True))
    cfg["strategies"]["volatile"]["enabled"] = st.sidebar.checkbox("Volatile Strategy", value=cfg["strategies"]["volatile"].get("enabled", True))
    cfg["strategies"]["cdc"]["enabled"]      = st.sidebar.checkbox("CDC Strategy",      value=cfg["strategies"]["cdc"].get("enabled", False))

    # ---- Global risk ----
    st.sidebar.header("Global Risk")
    cfg["risk"]["atr_stop_mult"] = st.sidebar.slider("ATR Stop Mult",  0.5, 5.0, float(cfg["risk"]["atr_stop_mult"]), 0.1)
    cfg["risk"]["atr_tp_mult"]   = st.sidebar.slider("ATR TP Mult",    0.5, 8.0, float(cfg["risk"]["atr_tp_mult"]),   0.1)
    cfg["hmm"]["min_confidence"] = st.sidebar.slider("Min Confidence", 0.3, 0.9, float(cfg["hmm"]["min_confidence"]), 0.05)

    # ---- Per-strategy params ----
    with st.sidebar.expander("Trend Params"):
        cfg["strategies"]["trend"]["min_prob"]      = st.slider("Min Prob",      0.3, 0.9, float(cfg["strategies"]["trend"]["min_prob"]),      0.05, key="trend_min_prob")
        cfg["strategies"]["trend"]["adx_threshold"] = st.slider("ADX Threshold", 5,  40,  int(cfg["strategies"]["trend"]["adx_threshold"]),    1,    key="trend_adx")
        cfg["strategies"]["trend"]["sizing_base"]   = st.slider("Sizing Base",   0.1, 1.0, float(cfg["strategies"]["trend"]["sizing_base"]),   0.05, key="trend_sz_base")
        cfg["strategies"]["trend"]["sizing_max"]    = st.slider("Sizing Max",    0.1, 2.0, float(cfg["strategies"]["trend"]["sizing_max"]),    0.05, key="trend_sz_max")

    with st.sidebar.expander("Range Params"):
        cfg["strategies"]["range"]["min_prob"]      = st.slider("Min Prob",      0.3, 0.9, float(cfg["strategies"]["range"]["min_prob"]),      0.05, key="range_min_prob")
        cfg["strategies"]["range"]["adx_threshold"] = st.slider("ADX Threshold", 5,  40,  int(cfg["strategies"]["range"]["adx_threshold"]),    1,    key="range_adx")
        cfg["strategies"]["range"]["sizing_base"]   = st.slider("Sizing Base",   0.1, 1.0, float(cfg["strategies"]["range"]["sizing_base"]),   0.05, key="range_sz_base")

    with st.sidebar.expander("CDC Params"):
        cfg["strategies"]["cdc"]["fast_period"]   = st.slider("Fast EMA",      5,  30, int(cfg["strategies"]["cdc"]["fast_period"]),    1,   key="cdc_fast")
        cfg["strategies"]["cdc"]["slow_period"]   = st.slider("Slow EMA",      10, 60, int(cfg["strategies"]["cdc"]["slow_period"]),    1,   key="cdc_slow")
        cfg["strategies"]["cdc"]["adx_threshold"] = st.slider("ADX Threshold", 5,  40, float(cfg["strategies"]["cdc"]["adx_threshold"]), 1.0, key="cdc_adx")
        cfg["strategies"]["cdc"]["sizing_base"]   = st.slider("Sizing Base",   0.1, 1.0, float(cfg["strategies"]["cdc"]["sizing_base"]),  0.05, key="cdc_sz_base")
        cfg["strategies"]["cdc"]["regime_gated"]  = st.checkbox("Regime Gated", value=cfg["strategies"]["cdc"]["regime_gated"], key="cdc_gated")

    # ---- Action buttons ----
    st.sidebar.divider()
    run_clicked  = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)
    save_clicked = st.sidebar.button("Save for Comparison", use_container_width=True)

    cfg["_ui_run"]  = run_clicked
    cfg["_ui_save"] = save_clicked

    return cfg
