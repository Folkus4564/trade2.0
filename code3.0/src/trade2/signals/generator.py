"""
signals/generator.py - Signal generation for single-TF and multi-TF modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


_LONDON_HOURS = set(range(7, 17))
_NY_HOURS     = set(range(13, 22))
_ACTIVE_HOURS = _LONDON_HOURS | _NY_HOURS


def _session_mask(index: pd.DatetimeIndex, allowed_hours: set) -> pd.Series:
    hours = index.hour if index.tz is None else index.tz_convert("UTC").hour
    return pd.Series(hours.isin(allowed_hours), index=index)


def generate_signals(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    # Multi-TF mode: regime already in df (from forward_fill_1h_regime)
    # Single-TF mode: pass these explicitly
    hmm_labels: pd.Series = None,
    hmm_bull_prob: np.ndarray = None,
    hmm_bear_prob: np.ndarray = None,
    hmm_index: pd.Index = None,
    # Overrides (for optimization)
    adx_threshold: float = None,
    hmm_min_prob: float = None,
    regime_persistence_bars: int = None,
    require_smc_confluence: bool = None,
    require_pin_bar: bool = None,
    atr_expansion_filter: bool = False,
    session_filter: bool = None,
    hmm_min_confidence: float = None,
    transition_cooldown_bars: int = None,
) -> pd.DataFrame:
    """
    Generate long/short entry and exit signals.

    Supports two modes:
    - multi_tf: regime/bull_prob/bear_prob columns already in df (from forward_fill)
    - single_tf: hmm_labels + hmm_bull_prob + hmm_bear_prob + hmm_index passed explicitly

    Entry logic:
    - Tiered SMC confluence (if require_smc_confluence):
        Tier 1: sweep + (OB or FVG)
        Tier 2: OB + FVG overlap
    - OR logic (if not require_smc_confluence): any SMC signal
    - Plus Donchian breakout fallback
    - Regime persistence filter (N bars stable before entry)

    Returns:
        df with signal_long, signal_short, exit_long, exit_short,
        position_size_long, position_size_short columns.
    """
    cfg      = config or {}
    hmm_cfg  = cfg["hmm"]
    reg_cfg  = cfg["regime"]
    smc_cfg  = cfg.get("smc_5m") or cfg["smc"]
    sess_cfg = cfg["session"]

    # Resolve parameters (explicit overrides > config)
    adx_thresh    = adx_threshold           if adx_threshold           is not None else reg_cfg["adx_threshold"]
    min_prob      = hmm_min_prob            if hmm_min_prob            is not None else hmm_cfg["min_prob_hard"]
    # Shorts require higher conviction — use min_prob_short if defined, else same as longs
    min_prob_short = hmm_cfg.get("min_prob_hard_short", min_prob)
    persistence   = regime_persistence_bars if regime_persistence_bars is not None else reg_cfg["persistence_bars"]
    confluence    = require_smc_confluence  if require_smc_confluence  is not None else smc_cfg["require_confluence"]
    pin_bar       = require_pin_bar         if require_pin_bar         is not None else smc_cfg["require_pin_bar"]
    sizing_base   = hmm_cfg["sizing_base"]
    sizing_max    = hmm_cfg["sizing_max"]
    sess_enabled  = session_filter if session_filter is not None else sess_cfg["enabled"]
    allowed_hours = set(sess_cfg.get("allowed_hours_utc", list(_ACTIVE_HOURS)))
    min_confidence = hmm_min_confidence if hmm_min_confidence is not None else hmm_cfg["min_confidence"]
    cooldown       = transition_cooldown_bars if transition_cooldown_bars is not None else reg_cfg["transition_cooldown_bars"]

    out = df.copy()

    # ---- Align regime columns ----
    if "regime" not in out.columns:
        # Single-TF mode: attach HMM outputs by index alignment
        if hmm_labels is None:
            raise ValueError("Either pass regime in df (multi_tf) or provide hmm_labels (single_tf)")
        regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
        bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
        bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")
        out["regime"]    = regime_s.reindex(out.index).fillna("sideways")
        out["bull_prob"] = bull_prob_s.reindex(out.index).fillna(0.0)
        out["bear_prob"] = bear_prob_s.reindex(out.index).fillna(0.0)

    # ---- Sideways probability ----
    if "sideways_prob" not in out.columns:
        # Derive from complement (works for both 2-state and 3-state HMM)
        out["sideways_prob"] = (1.0 - out["bull_prob"] - out["bear_prob"]).clip(lower=0.0)

    # ---- Uncertainty filter: max regime prob must exceed min_confidence ----
    max_prob  = out[["bull_prob", "bear_prob", "sideways_prob"]].max(axis=1)
    confident = max_prob >= min_confidence

    # ---- Regime direction: probability-only gating (no hard label requirement) ----
    bull_raw = confident & (out["bull_prob"] >= min_prob)
    bear_raw = confident & (out["bear_prob"] >= min_prob_short)

    # ---- Transition cooldown: suppress entries N bars after regime flip ----
    if cooldown > 0:
        freq_mult    = 12 if _is_5m_data(out) else 1
        cooldown_bars = cooldown * freq_mult
        regime_changed = out["regime"] != out["regime"].shift(1)
        in_cooldown    = regime_changed.rolling(cooldown_bars, min_periods=1).sum() > 0
        bull_raw = bull_raw & ~in_cooldown
        bear_raw = bear_raw & ~in_cooldown

    # Shorts require longer persistence (more bars of confirmed bear regime).
    # Config: regime.persistence_bars_short; defaults to persistence if not set.
    persistence_short = reg_cfg.get("persistence_bars_short", persistence)

    if persistence > 1 or persistence_short > 1:
        # For multi-TF 5M: persistence is in 1H bars, so x12 on 5M
        freq_mult = 12 if _is_5m_data(out) else 1
        win_long  = max(persistence,       1) * freq_mult
        win_short = max(persistence_short, 1) * freq_mult
        bull_regime = bull_raw.rolling(win_long).sum()  == win_long
        bear_regime = bear_raw.rolling(win_short).sum() == win_short
    else:
        bull_regime = bull_raw
        bear_regime = bear_raw

    # ---- SMC entry signals ----
    has_smc = all(c in out.columns for c in ["ob_bullish","ob_bearish","fvg_bullish","fvg_bearish","sweep_low","sweep_high"])
    has_pin = "pin_bar_bull" in out.columns and "pin_bar_bear" in out.columns

    if has_smc:
        if confluence:
            t1_long  = out["sweep_low"]  & (out["ob_bullish"] | out["fvg_bullish"])
            t1_short = out["sweep_high"] & (out["ob_bearish"] | out["fvg_bearish"])
            t2_long  = out["ob_bullish"] & out["fvg_bullish"]
            t2_short = out["ob_bearish"] & out["fvg_bearish"]
            smc_long  = t1_long  | t2_long
            smc_short = t1_short | t2_short
        else:
            smc_long  = out["ob_bullish"] | out["fvg_bullish"] | out["sweep_low"]
            smc_short = out["ob_bearish"] | out["fvg_bearish"] | out["sweep_high"]

        if pin_bar and has_pin:
            smc_long  = smc_long  & out["pin_bar_bull"]
            smc_short = smc_short & out["pin_bar_bear"]
    else:
        smc_long  = pd.Series(False, index=out.index)
        smc_short = pd.Series(False, index=out.index)

    # ---- Donchian breakout fallback ----
    has_dc  = "breakout_long" in out.columns and "adx_14" in out.columns
    atr_exp = out["atr_expansion"] == 1 if (atr_expansion_filter and "atr_expansion" in out.columns) else pd.Series(True, index=out.index)

    if has_dc:
        adx_ok   = out["adx_14"] > adx_thresh
        dc_long  = out["breakout_long"].astype(bool)  & adx_ok & atr_exp
        dc_short = out["breakout_short"].astype(bool) & adx_ok & atr_exp
    else:
        dc_long  = pd.Series(False, index=out.index)
        dc_short = pd.Series(False, index=out.index)

    # ---- Session filter ----
    if sess_enabled:
        in_session = _session_mask(out.index, allowed_hours)
        smc_long   = smc_long  & in_session
        smc_short  = smc_short & in_session
        dc_long    = dc_long   & in_session
        dc_short   = dc_short  & in_session

    # ---- Macro trend filter (optional) ----
    # If 1H HMA direction columns are present, require them to agree with direction.
    # Longs: HMA rising AND price above HMA (confirmed uptrend).
    # Shorts: HMA falling AND price below HMA (confirmed downtrend).
    has_macro = "hma_rising" in out.columns and "price_above_hma" in out.columns
    if has_macro:
        macro_bull = out["hma_rising"].astype(bool) & out["price_above_hma"].astype(bool)
        macro_bear = (~out["hma_rising"].astype(bool)) & (~out["price_above_hma"].astype(bool))
    else:
        macro_bull = pd.Series(True, index=out.index)
        macro_bear = pd.Series(True, index=out.index)

    # ---- Combine signals ----
    out["signal_long"]  = (bull_regime & macro_bull & (smc_long  | dc_long)).astype(int)
    out["signal_short"] = (bear_regime & macro_bear & (smc_short | dc_short)).astype(int)

    # ---- Exit: regime flip ----
    out["exit_long"]  = (~bull_regime).astype(int)
    out["exit_short"] = (~bear_regime).astype(int)

    # ---- Confidence-based position sizing (linear interpolation) ----
    # sizing_base at min_confidence -> sizing_max at prob=1.0
    prob_range   = max(1.0 - min_confidence, 1e-9)
    long_excess  = np.clip(out["bull_prob"] - min_confidence, 0.0, prob_range) / prob_range
    short_excess = np.clip(out["bear_prob"] - min_confidence, 0.0, prob_range) / prob_range
    size_long    = sizing_base + long_excess  * (sizing_max - sizing_base)
    size_short   = sizing_base + short_excess * (sizing_max - sizing_base)

    out["position_size_long"]  = np.where(out["signal_long"]  == 1, size_long,  0.0)
    out["position_size_short"] = np.where(out["signal_short"] == 1, size_short, 0.0)

    return out


def compute_stops(
    df: pd.DataFrame,
    atr_stop_mult: float,
    atr_tp_mult: float,
) -> pd.DataFrame:
    """
    Compute ATR-based stop loss and take profit levels.

    In multi-TF mode (5M signals, 1H regime), uses 'atr_1h' if present.
    This prevents the 5M noise from immediately triggering SL/TP.
    Falls back to 'atr_14' for single-TF mode.
    """
    out   = df.copy()
    atr   = out["atr_1h"] if "atr_1h" in out.columns else out["atr_14"]
    close = out["Close"]
    out["stop_long"]  = close - atr_stop_mult * atr
    out["stop_short"] = close + atr_stop_mult * atr
    out["tp_long"]    = close + atr_tp_mult   * atr
    out["tp_short"]   = close - atr_tp_mult   * atr
    return out


def compute_stops_regime_aware(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute ATR-based SL/TP levels with per-regime multipliers.

    Reads the 'signal_source' column (set by router: 'trend', 'range', 'volatile').
    For trend strategy with atr_tp_mult == 0: sets TP to a far-away level (ride regime).
    Untagged bars fall back to global risk.atr_stop_mult / risk.atr_tp_mult.
    """
    out   = df.copy()
    atr   = out["atr_1h"] if "atr_1h" in out.columns else out["atr_14"]
    close = out["Close"]

    strat_cfg = config["strategies"]
    risk_cfg  = config["risk"]

    default_stop_mult = risk_cfg["atr_stop_mult"]
    default_tp_mult   = risk_cfg["atr_tp_mult"]

    source = out["signal_source"] if "signal_source" in out.columns else pd.Series("", index=out.index)

    # Build per-bar multiplier arrays
    stop_mults = np.full(len(out), default_stop_mult, dtype=float)
    tp_mults   = np.full(len(out), default_tp_mult,   dtype=float)

    for src_name in ("trend", "range", "volatile", "cdc"):
        mask = (source == src_name).values
        if mask.any():
            scfg = strat_cfg[src_name]
            stop_mults[mask] = scfg["atr_stop_mult"]
            tp_mults[mask]   = scfg["atr_tp_mult"]

    # atr_tp_mult == 0 -> no fixed TP (ride regime); use a far-away level
    no_fixed_tp = tp_mults == 0.0
    tp_mults    = np.where(no_fixed_tp, 999.0, tp_mults)

    atr_vals = atr.values
    cl_vals  = close.values

    out["stop_long"]  = cl_vals - stop_mults * atr_vals
    out["stop_short"] = cl_vals + stop_mults * atr_vals
    out["tp_long"]    = cl_vals + tp_mults   * atr_vals
    out["tp_short"]   = cl_vals - tp_mults   * atr_vals

    # ---- Trailing stop multipliers (per-bar, 0 = disabled) ----
    # Consumed by _simulate_trades() to update frozen_sl dynamically.
    trail_mults = np.zeros(len(out), dtype=float)
    for src_name in ("trend", "range", "volatile", "cdc"):
        mask = (source == src_name).values
        if mask.any():
            scfg = strat_cfg.get(src_name, {})
            if scfg.get("trailing_enabled", False):
                trail_mults[mask] = float(scfg["trailing_atr_mult"])

    out["trailing_atr_mult_long"]  = trail_mults
    out["trailing_atr_mult_short"] = trail_mults

    return out


def _is_5m_data(df: pd.DataFrame) -> bool:
    """Heuristic: check if DataFrame has 5-minute frequency."""
    if len(df) < 2:
        return False
    try:
        delta = df.index[1] - df.index[0]
        return delta.total_seconds() <= 360  # <= 6 minutes = 5M
    except Exception:
        return False
