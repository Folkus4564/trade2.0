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
    cfg     = config or {}
    hmm_cfg = cfg.get("hmm", {})
    reg_cfg = cfg.get("regime", {})
    smc_cfg = cfg.get("smc", cfg.get("smc_5m", {}))
    sess_cfg = cfg.get("session", {})

    # Resolve parameters (explicit overrides > config > defaults)
    adx_thresh    = adx_threshold            if adx_threshold            is not None else reg_cfg.get("adx_threshold",  20.0)
    min_prob      = hmm_min_prob             if hmm_min_prob             is not None else hmm_cfg.get("min_prob_hard",   0.50)
    persistence   = regime_persistence_bars  if regime_persistence_bars  is not None else reg_cfg.get("persistence_bars", 3)
    confluence    = require_smc_confluence   if require_smc_confluence   is not None else smc_cfg.get("require_confluence", True)
    pin_bar       = require_pin_bar          if require_pin_bar          is not None else smc_cfg.get("require_pin_bar",    False)
    sizing_base   = hmm_cfg.get("sizing_base", 0.50)
    sizing_max    = hmm_cfg.get("sizing_max",  1.50)
    sess_enabled  = session_filter if session_filter is not None else sess_cfg.get("enabled", False)
    allowed_hours = set(sess_cfg.get("allowed_hours_utc", list(_ACTIVE_HOURS)))

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

    # ---- Regime filter + persistence ----
    bull_raw = (out["regime"] == "bull") & (out["bull_prob"] >= min_prob)
    bear_raw = (out["regime"] == "bear") & (out["bear_prob"] >= min_prob)

    if persistence > 1:
        # For multi-TF 5M: persistence is in 1H bars, so x12 on 5M
        # Detect if this is 5M data by checking index freq or len ratio
        freq_mult = 12 if _is_5m_data(out) else 1
        win = persistence * freq_mult
        bull_regime = bull_raw.rolling(win).sum() == win
        bear_regime = bear_raw.rolling(win).sum() == win
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

    # ---- Combine signals ----
    out["signal_long"]  = (bull_regime & (smc_long  | dc_long)).astype(int)
    out["signal_short"] = (bear_regime & (smc_short | dc_short)).astype(int)

    # ---- Exit: regime flip ----
    out["exit_long"]  = (~bull_regime).astype(int)
    out["exit_short"] = (~bear_regime).astype(int)

    # ---- Confidence-based position sizing ----
    out["position_size_long"]  = np.where(
        out["signal_long"]  == 1, np.clip(sizing_base + out["bull_prob"], 0.1, sizing_max), 0.0
    )
    out["position_size_short"] = np.where(
        out["signal_short"] == 1, np.clip(sizing_base + out["bear_prob"], 0.1, sizing_max), 0.0
    )

    return out


def compute_stops(
    df: pd.DataFrame,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
) -> pd.DataFrame:
    """Compute ATR-based stop loss and take profit levels."""
    out   = df.copy()
    atr   = out["atr_14"]
    close = out["Close"]
    out["stop_long"]  = close - atr_stop_mult * atr
    out["stop_short"] = close + atr_stop_mult * atr
    out["tp_long"]    = close + atr_tp_mult   * atr
    out["tp_short"]   = close - atr_tp_mult   * atr
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
