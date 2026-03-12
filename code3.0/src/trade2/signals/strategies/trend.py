"""
signals/strategies/trend.py - Trend-following sub-strategy.

Entry logic: Donchian breakout (primary) + optional SMC confirm + macro trend filter.
Regime: bull_prob >= min_prob (bullish) or bear_prob >= min_prob (bearish).
Exit: regime flip (no fixed TP when atr_tp_mult == 0).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def trend_strategy(
    df: pd.DataFrame,
    config: Dict[str, Any],
    long_mask: pd.Series,
    short_mask: pd.Series,
) -> pd.DataFrame:
    """
    Trend-following strategy for bull/bear regime bars.

    Args:
        df:         Signal DataFrame with all 5M features + regime columns.
        config:     Full config dict.
        long_mask:  Boolean mask where bull_prob >= trend.min_prob.
        short_mask: Boolean mask where bear_prob >= trend.min_prob.

    Returns:
        DataFrame with signal_long, signal_short, exit_long, exit_short,
        position_size_long, position_size_short, signal_source columns.
    """
    from trade2.signals.generator import _session_mask, _is_5m_data

    tcfg = config["strategies"]["trend"]
    reg_cfg = config["regime"]

    out = df.copy()
    n = len(out)

    # ---- Donchian breakout ----
    has_dc = "breakout_long" in out.columns and "adx_14" in out.columns
    adx_thresh = tcfg["adx_threshold"]
    atr_exp_filter = tcfg["atr_expansion_filter"]
    atr_exp = (out["atr_expansion"] == 1) if (atr_exp_filter and "atr_expansion" in out.columns) \
              else pd.Series(True, index=out.index)

    if has_dc:
        adx_ok   = out["adx_14"] > adx_thresh
        dc_long  = out["breakout_long"].astype(bool)  & adx_ok & atr_exp
        dc_short = out["breakout_short"].astype(bool) & adx_ok & atr_exp
    else:
        dc_long  = pd.Series(False, index=out.index)
        dc_short = pd.Series(False, index=out.index)

    # ---- Optional SMC confirmation ----
    if tcfg["require_smc_confirm"]:
        has_smc = all(c in out.columns for c in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish"])
        if has_smc:
            smc_long  = out["ob_bullish"] | out["fvg_bullish"]
            smc_short = out["ob_bearish"] | out["fvg_bearish"]
        else:
            smc_long  = pd.Series(True, index=out.index)
            smc_short = pd.Series(True, index=out.index)
    else:
        smc_long  = pd.Series(True, index=out.index)
        smc_short = pd.Series(True, index=out.index)

    # ---- Macro trend filter ----
    if tcfg["require_macro_trend"] and "hma_rising" in out.columns and "price_above_hma" in out.columns:
        macro_bull = out["hma_rising"].astype(bool) & out["price_above_hma"].astype(bool)
        macro_bear = (~out["hma_rising"].astype(bool)) & (~out["price_above_hma"].astype(bool))
    else:
        macro_bull = pd.Series(True, index=out.index)
        macro_bear = pd.Series(True, index=out.index)

    # ---- Regime persistence ----
    persistence = tcfg["persistence_bars"]
    if persistence > 1:
        freq_mult  = 12 if _is_5m_data(out) else 1
        win        = max(persistence, 1) * freq_mult
        bull_regime = long_mask.rolling(win).sum()  == win
        bear_regime = short_mask.rolling(win).sum() == win
    else:
        bull_regime = long_mask
        bear_regime = short_mask

    # ---- Session filter ----
    if tcfg["session_enabled"]:
        allowed = set(tcfg["allowed_hours_utc"])
        in_sess  = _session_mask(out.index, allowed)
        dc_long  = dc_long  & in_sess
        dc_short = dc_short & in_sess

    # ---- Combine ----
    sig_long  = (bull_regime & macro_bull & dc_long  & smc_long).astype(int)
    sig_short = (bear_regime & macro_bear & dc_short & smc_short).astype(int)

    # ---- Exit: regime flip ----
    exit_long  = (~bull_regime).astype(int)
    exit_short = (~bear_regime).astype(int)

    # ---- Confidence-based sizing ----
    min_prob   = tcfg["min_prob"]
    prob_range = max(1.0 - min_prob, 1e-9)
    sz_base    = tcfg["sizing_base"]
    sz_max     = tcfg["sizing_max"]

    long_excess  = np.clip(out["bull_prob"] - min_prob, 0.0, prob_range) / prob_range
    short_excess = np.clip(out["bear_prob"] - min_prob, 0.0, prob_range) / prob_range
    size_long    = sz_base + long_excess  * (sz_max - sz_base)
    size_short   = sz_base + short_excess * (sz_max - sz_base)

    out["signal_long"]          = sig_long
    out["signal_short"]         = sig_short
    out["exit_long"]            = exit_long
    out["exit_short"]           = exit_short
    out["position_size_long"]   = np.where(sig_long  == 1, size_long,  0.0)
    out["position_size_short"]  = np.where(sig_short == 1, size_short, 0.0)
    # Tag source only on signal bars; fill rest with empty string
    out["signal_source"] = np.where((sig_long == 1) | (sig_short == 1), "trend", "")

    return out
