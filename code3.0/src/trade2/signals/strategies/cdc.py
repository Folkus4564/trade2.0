"""
signals/strategies/cdc.py - CDC Action Zone sub-strategy.

Entry: cdc_buy signal (first bar entering green zone).
Exit:  cdc_sell signal (first bar entering blue zone).
Regime gating: optional (regime_gated: true = trend-regime bars only).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def cdc_strategy(
    df: pd.DataFrame,
    config: Dict[str, Any],
    long_mask: pd.Series,
    short_mask: pd.Series,
) -> pd.DataFrame:
    """
    CDC Action Zone sub-strategy.

    Args:
        df:          Signal DataFrame with CDC feature columns.
        config:      Full config dict.
        long_mask:   Boolean mask for allowed long bars (regime gate or all-True).
        short_mask:  Boolean mask for allowed short bars (regime gate or all-True).

    Returns:
        DataFrame with 7 standard signal columns.
    """
    from trade2.signals.generator import _session_mask

    cdc_cfg = config["strategies"]["cdc"]

    out = df.copy()

    # ---- Check required columns ----
    has_cdc = "cdc_buy" in out.columns and "cdc_sell" in out.columns
    if not has_cdc:
        out["signal_long"]         = 0
        out["signal_short"]        = 0
        out["exit_long"]           = 1
        out["exit_short"]          = 1
        out["position_size_long"]  = 0.0
        out["position_size_short"] = 0.0
        out["signal_source"]       = ""
        return out

    # ---- Base entry signals ----
    cdc_buy  = out["cdc_buy"].astype(bool)
    cdc_sell = out["cdc_sell"].astype(bool)

    # ---- BOS confirmation (optional) ----
    if cdc_cfg["require_bos_confirm"]:
        has_bos = "bos_bullish" in out.columns and "bos_bearish" in out.columns
        if has_bos:
            cdc_buy  = cdc_buy  & out["bos_bullish"].astype(bool)
            cdc_sell = cdc_sell & out["bos_bearish"].astype(bool)

    # ---- ADX filter ----
    adx_thresh = cdc_cfg["adx_threshold"]
    if "adx_14" in out.columns and adx_thresh > 0:
        adx_ok = out["adx_14"] > adx_thresh
        cdc_buy  = cdc_buy  & adx_ok
        cdc_sell = cdc_sell & adx_ok

    # ---- Session filter ----
    if cdc_cfg["session_enabled"]:
        allowed = set(cdc_cfg["allowed_hours_utc"])
        in_sess  = _session_mask(out.index, allowed)
        cdc_buy  = cdc_buy  & in_sess
        cdc_sell = cdc_sell & in_sess

    # ---- Regime gating ----
    if cdc_cfg["regime_gated"]:
        sig_long  = (long_mask  & cdc_buy).astype(int)
        sig_short = (short_mask & cdc_sell).astype(int)
    else:
        sig_long  = cdc_buy.astype(int)
        sig_short = cdc_sell.astype(int)

    # ---- Exit: opposing CDC signal ----
    exit_long  = cdc_sell.astype(int)
    exit_short = cdc_buy.astype(int)

    # ---- Confidence-based sizing ----
    sz_base = cdc_cfg["sizing_base"]
    sz_max  = cdc_cfg["sizing_max"]

    if cdc_cfg["regime_gated"] and "bull_prob" in out.columns:
        min_prob   = config["strategies"]["trend"]["min_prob"]
        prob_range = max(1.0 - min_prob, 1e-9)
        long_excess  = np.clip(out["bull_prob"] - min_prob, 0.0, prob_range) / prob_range
        short_excess = np.clip(out["bear_prob"] - min_prob, 0.0, prob_range) / prob_range
        size_long    = sz_base + long_excess  * (sz_max - sz_base)
        size_short   = sz_base + short_excess * (sz_max - sz_base)
    else:
        size_long  = pd.Series(sz_base, index=out.index)
        size_short = pd.Series(sz_base, index=out.index)

    out["signal_long"]         = sig_long
    out["signal_short"]        = sig_short
    out["exit_long"]           = exit_long
    out["exit_short"]          = exit_short
    out["position_size_long"]  = np.where(sig_long  == 1, size_long,  0.0)
    out["position_size_short"] = np.where(sig_short == 1, size_short, 0.0)
    out["signal_source"]       = np.where((sig_long == 1) | (sig_short == 1), "cdc", "")

    return out
