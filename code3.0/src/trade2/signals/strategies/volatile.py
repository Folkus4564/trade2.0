"""
signals/strategies/volatile.py - Ultra-selective SMC strategy for uncertain regime bars.

Entry: triple SMC confluence (sweep + OB + FVG) + optional pin bar + high ADX.
Regime: max(probs) < volatile.max_confidence (no dominant regime).
Exit: tight TP/SL only (1.0x stop, 1.5x TP) with 12-bar timeout.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def volatile_strategy(
    df: pd.DataFrame,
    config: Dict[str, Any],
    volatile_mask: pd.Series,
) -> pd.DataFrame:
    """
    Ultra-selective SMC strategy for volatile/uncertain regime bars.

    If volatile.enabled == false, returns zero signals for all bars.

    Args:
        df:             Signal DataFrame with all 5M features + regime columns.
        config:         Full config dict.
        volatile_mask:  Boolean mask where max(probs) < max_confidence.

    Returns:
        DataFrame with signal_long, signal_short, exit_long, exit_short,
        position_size_long, position_size_short, signal_source columns.
    """
    from trade2.signals.generator import _session_mask

    vcfg = config["strategies"]["volatile"]

    out = df.copy()
    zeros = pd.Series(0, index=out.index)
    empty = pd.Series("", index=out.index)

    # Disabled: skip all uncertain bars
    if not vcfg["enabled"]:
        out["signal_long"]         = zeros
        out["signal_short"]        = zeros
        out["exit_long"]           = zeros
        out["exit_short"]          = zeros
        out["position_size_long"]  = 0.0
        out["position_size_short"] = 0.0
        out["signal_source"]       = empty
        return out

    # ---- Triple SMC confluence ----
    has_smc = all(c in out.columns for c in
                  ["sweep_low", "sweep_high", "ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish"])

    if has_smc:
        triple_long  = out["sweep_low"]  & out["ob_bullish"] & out["fvg_bullish"]
        triple_short = out["sweep_high"] & out["ob_bearish"] & out["fvg_bearish"]
    else:
        triple_long  = pd.Series(False, index=out.index)
        triple_short = pd.Series(False, index=out.index)

    # ---- Pin bar filter ----
    if vcfg["require_pin_bar"]:
        has_pin = "pin_bar_bull" in out.columns and "pin_bar_bear" in out.columns
        if has_pin:
            triple_long  = triple_long  & out["pin_bar_bull"].astype(bool)
            triple_short = triple_short & out["pin_bar_bear"].astype(bool)

    # ---- ADX filter (confirm momentum despite uncertainty) ----
    has_adx = "adx_14" in out.columns
    if has_adx:
        adx_ok = out["adx_14"] > vcfg["adx_threshold"]
    else:
        adx_ok = pd.Series(True, index=out.index)

    # ---- Session filter ----
    if vcfg["session_enabled"]:
        allowed = set(vcfg["allowed_hours_utc"])
        in_sess  = _session_mask(out.index, allowed)
        triple_long  = triple_long  & in_sess
        triple_short = triple_short & in_sess

    # ---- Combine ----
    sig_long  = (volatile_mask & triple_long  & adx_ok).astype(int)
    sig_short = (volatile_mask & triple_short & adx_ok).astype(int)

    exit_long  = (~volatile_mask).astype(int)
    exit_short = (~volatile_mask).astype(int)

    # Minimal sizing
    sz_base  = vcfg["sizing_base"]
    sz_max   = vcfg["sizing_max"]
    mid_size = (sz_base + sz_max) / 2.0   # flat sizing for uncertain regime

    out["signal_long"]          = sig_long
    out["signal_short"]         = sig_short
    out["exit_long"]            = exit_long
    out["exit_short"]           = exit_short
    out["position_size_long"]   = np.where(sig_long  == 1, mid_size, 0.0)
    out["position_size_short"]  = np.where(sig_short == 1, mid_size, 0.0)
    out["signal_source"]        = np.where((sig_long == 1) | (sig_short == 1), "volatile", "")

    return out
