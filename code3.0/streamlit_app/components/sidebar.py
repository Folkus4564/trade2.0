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

    # ---- Strategy mode ----
    st.sidebar.header("Strategy Mode")
    mode = st.sidebar.radio(
        "Signal Mode",
        ["legacy", "regime_specialized"],
        index=0,
        help="legacy = proven 12.86% base | regime_specialized = trend/range/volatile router",
    )
    cfg["strategies"]["mode"] = mode

    # ---- Strategy toggles ----
    st.sidebar.header("Strategy Toggles")
    cfg["strategies"]["trend"]["enabled"]    = st.sidebar.checkbox("Trend Strategy",    value=bool(cfg["strategies"]["trend"].get("enabled", True)))
    cfg["strategies"]["range"]["enabled"]    = st.sidebar.checkbox("Range Strategy",    value=bool(cfg["strategies"]["range"].get("enabled", True)))
    cfg["strategies"]["volatile"]["enabled"] = st.sidebar.checkbox("Volatile Strategy", value=bool(cfg["strategies"]["volatile"].get("enabled", True)))
    cfg["strategies"]["cdc"]["enabled"]      = st.sidebar.checkbox("CDC Strategy",      value=bool(cfg["strategies"]["cdc"].get("enabled", False)))

    # ---- Global risk ----
    st.sidebar.header("Global Risk")
    cfg["risk"]["atr_stop_mult"] = st.sidebar.slider("ATR Stop Mult",  0.5, 5.0, float(cfg["risk"]["atr_stop_mult"]), 0.1)
    cfg["risk"]["atr_tp_mult"]   = st.sidebar.slider("ATR TP Mult",    0.5, 8.0, float(cfg["risk"]["atr_tp_mult"]),   0.1)
    cfg["hmm"]["min_confidence"] = st.sidebar.slider("Min Confidence", 0.3, 0.9, float(cfg["hmm"]["min_confidence"]), 0.05)

    # ---- Per-strategy params ----
    # All sliders use float types — YAML loads 15.0 as float, mixing int/float
    # causes StreamlitAPIException on slider step type mismatch.
    with st.sidebar.expander("Trend Params"):
        cfg["strategies"]["trend"]["min_prob"]      = st.slider("Min Prob",      0.30, 0.90, float(cfg["strategies"]["trend"]["min_prob"]),      0.05, key="trend_min_prob")
        cfg["strategies"]["trend"]["adx_threshold"] = st.slider("ADX Threshold", 5.0,  40.0, float(cfg["strategies"]["trend"]["adx_threshold"]), 1.0,  key="trend_adx")
        cfg["strategies"]["trend"]["sizing_base"]   = st.slider("Sizing Base",   0.10, 1.00, float(cfg["strategies"]["trend"]["sizing_base"]),   0.05, key="trend_sz_base")
        cfg["strategies"]["trend"]["sizing_max"]    = st.slider("Sizing Max",    0.10, 2.00, float(cfg["strategies"]["trend"]["sizing_max"]),    0.05, key="trend_sz_max")
        cfg["strategies"]["trend"]["require_bos_confirm"] = st.checkbox(
            "Require BOS Confirm (LuxAlgo)",
            value=bool(cfg["strategies"]["trend"].get("require_bos_confirm", False)),
            key="trend_bos",
        )

    with st.sidebar.expander("Range Params"):
        cfg["strategies"]["range"]["min_prob"]    = st.slider("Min Prob",    0.30, 0.90, float(cfg["strategies"]["range"]["min_prob"]),    0.05, key="range_min_prob")
        cfg["strategies"]["range"]["adx_max"]     = st.slider("ADX Max",     5.0,  40.0, float(cfg["strategies"]["range"]["adx_max"]),     1.0,  key="range_adx")
        cfg["strategies"]["range"]["sizing_base"] = st.slider("Sizing Base", 0.10, 1.00, float(cfg["strategies"]["range"]["sizing_base"]), 0.05, key="range_sz_base")

    with st.sidebar.expander("Volatile Params"):
        cfg["strategies"]["volatile"]["max_confidence"] = st.slider(
            "Max Confidence", 0.30, 0.80, float(cfg["strategies"]["volatile"].get("max_confidence", 0.50)), 0.05, key="vol_max_conf"
        )
        cfg["strategies"]["volatile"]["adx_threshold"] = st.slider(
            "ADX Threshold", 5.0, 50.0, float(cfg["strategies"]["volatile"].get("adx_threshold", 30.0)), 1.0, key="vol_adx"
        )
        cfg["strategies"]["volatile"]["sizing_base"] = st.slider(
            "Sizing Base", 0.05, 0.50, float(cfg["strategies"]["volatile"].get("sizing_base", 0.15)), 0.05, key="vol_sz_base"
        )
        cfg["strategies"]["volatile"]["sizing_max"] = st.slider(
            "Sizing Max", 0.10, 1.00, float(cfg["strategies"]["volatile"].get("sizing_max", 0.30)), 0.05, key="vol_sz_max"
        )

    with st.sidebar.expander("CDC Params"):
        cfg["strategies"]["cdc"]["fast_period"]   = st.slider("Fast EMA",      5.0,  30.0, float(cfg["strategies"]["cdc"]["fast_period"]),   1.0,  key="cdc_fast")
        cfg["strategies"]["cdc"]["slow_period"]   = st.slider("Slow EMA",      10.0, 60.0, float(cfg["strategies"]["cdc"]["slow_period"]),   1.0,  key="cdc_slow")
        cfg["strategies"]["cdc"]["adx_threshold"] = st.slider("ADX Threshold", 5.0,  40.0, float(cfg["strategies"]["cdc"]["adx_threshold"]), 1.0,  key="cdc_adx")
        cfg["strategies"]["cdc"]["sizing_base"]   = st.slider("Sizing Base",   0.10, 1.00, float(cfg["strategies"]["cdc"]["sizing_base"]),   0.05, key="cdc_sz_base")
        cfg["strategies"]["cdc"]["regime_gated"]  = st.checkbox("Regime Gated", value=bool(cfg["strategies"]["cdc"].get("regime_gated", True)), key="cdc_gated")
        cfg["strategies"]["cdc"]["require_bos_confirm"] = st.checkbox(
            "Require BOS Confirm (LuxAlgo)",
            value=bool(cfg["strategies"]["cdc"].get("require_bos_confirm", False)),
            key="cdc_bos",
        )

    # ---- LuxAlgo SMC params ----
    with st.sidebar.expander("LuxAlgo SMC"):
        # setdefault guards against custom configs missing these sections
        cfg.setdefault("smc_luxalgo",    {"enabled": True, "swing_left_bars": 5,  "swing_right_bars": 5,  "equal_hl_atr_mult": 0.1})
        cfg.setdefault("smc_luxalgo_5m", {"enabled": True, "swing_left_bars": 10, "swing_right_bars": 10, "equal_hl_atr_mult": 0.1})

        smc   = cfg["smc_luxalgo"]
        smc5m = cfg["smc_luxalgo_5m"]

        smc["enabled"]          = st.checkbox("LuxAlgo Enabled (1H)", value=bool(smc.get("enabled", True)),  key="smc_enabled")
        smc["swing_left_bars"]  = st.slider("Swing Left Bars (1H)",  2, 15, int(smc.get("swing_left_bars",  5)),  1, key="smc_left")
        smc["swing_right_bars"] = st.slider("Swing Right Bars (1H)", 2, 15, int(smc.get("swing_right_bars", 5)),  1, key="smc_right")

        smc5m["enabled"]          = st.checkbox("LuxAlgo Enabled (5M)", value=bool(smc5m.get("enabled", True)), key="smc5m_enabled")
        smc5m["swing_left_bars"]  = st.slider("Swing Left Bars (5M)",  2, 20, int(smc5m.get("swing_left_bars",  10)), 1, key="smc5m_left")
        smc5m["swing_right_bars"] = st.slider("Swing Right Bars (5M)", 2, 20, int(smc5m.get("swing_right_bars", 10)), 1, key="smc5m_right")

    # ---- Action buttons ----
    st.sidebar.divider()
    run_clicked  = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)
    save_clicked = st.sidebar.button("Save for Comparison", use_container_width=True)

    # Auto-run on first load if no result exists yet
    cfg["_ui_run"]  = run_clicked or (st.session_state.get("current_run") is None)
    cfg["_ui_save"] = save_clicked

    return cfg
