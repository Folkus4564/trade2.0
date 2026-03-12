"""
signals/strategies/range.py - Mean-reversion sub-strategy.

Entry logic: BB position (near lower/upper band) + RSI oversold/overbought
             + optional SMC order-block confirmation.
Regime: sideways_prob >= range.min_prob.
Exit: regime flip OR tight TP/SL (1.5x stop, 2.5x TP).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def range_strategy(
    df: pd.DataFrame,
    config: Dict[str, Any],
    range_mask: pd.Series,
) -> pd.DataFrame:
    """
    Mean-reversion strategy for sideways regime bars.

    Args:
        df:          Signal DataFrame with all 5M features + regime columns.
        config:      Full config dict.
        range_mask:  Boolean mask where sideways_prob >= range.min_prob.

    Returns:
        DataFrame with signal_long, signal_short, exit_long, exit_short,
        position_size_long, position_size_short, signal_source columns.
    """
    from trade2.signals.generator import _session_mask, _is_5m_data

    rcfg = config["strategies"]["range"]

    out = df.copy()

    # ---- BB position filter ----
    has_bb = "bb_pos" in out.columns
    if has_bb:
        bb_long  = out["bb_pos"] < rcfg["bb_pos_long_max"]     # near lower BB
        bb_short = out["bb_pos"] > rcfg["bb_pos_short_min"]    # near upper BB
    else:
        bb_long  = pd.Series(False, index=out.index)
        bb_short = pd.Series(False, index=out.index)

    # ---- RSI filter ----
    has_rsi = "rsi_14" in out.columns
    if has_rsi:
        rsi_long  = out["rsi_14"] < rcfg["rsi_long_max"]       # oversold
        rsi_short = out["rsi_14"] > rcfg["rsi_short_min"]      # overbought
    else:
        rsi_long  = pd.Series(True, index=out.index)
        rsi_short = pd.Series(True, index=out.index)

    # ---- ADX confirm low-trend environment ----
    has_adx = "adx_14" in out.columns
    if has_adx:
        adx_low = out["adx_14"] < rcfg["adx_max"]
    else:
        adx_low = pd.Series(True, index=out.index)

    # ---- SMC order-block confirmation ----
    if rcfg["require_smc_ob"]:
        has_ob = "ob_bullish" in out.columns and "ob_bearish" in out.columns
        if has_ob:
            ob_long  = out["ob_bullish"].astype(bool)
            ob_short = out["ob_bearish"].astype(bool)
        else:
            ob_long  = pd.Series(True, index=out.index)
            ob_short = pd.Series(True, index=out.index)
    else:
        ob_long  = pd.Series(True, index=out.index)
        ob_short = pd.Series(True, index=out.index)

    # ---- Regime persistence ----
    persistence = rcfg["persistence_bars"]
    if persistence > 1:
        freq_mult    = 12 if _is_5m_data(out) else 1
        win          = max(persistence, 1) * freq_mult
        range_regime = range_mask.rolling(win).sum() == win
    else:
        range_regime = range_mask

    # ---- Session filter ----
    if rcfg["session_enabled"]:
        allowed  = set(rcfg["allowed_hours_utc"])
        in_sess  = _session_mask(out.index, allowed)
        bb_long  = bb_long  & in_sess
        bb_short = bb_short & in_sess

    # ---- Combine ----
    sig_long  = (range_regime & bb_long  & rsi_long  & adx_low & ob_long).astype(int)
    sig_short = (range_regime & bb_short & rsi_short & adx_low & ob_short).astype(int)

    # Exit: regime no longer sideways
    exit_long  = (~range_regime).astype(int)
    exit_short = (~range_regime).astype(int)

    # ---- Confidence-based sizing ----
    min_prob   = rcfg["min_prob"]
    prob_range = max(1.0 - min_prob, 1e-9)
    sz_base    = rcfg["sizing_base"]
    sz_max     = rcfg["sizing_max"]

    sw_prob = out["sideways_prob"] if "sideways_prob" in out.columns else pd.Series(0.0, index=out.index)
    sw_excess = np.clip(sw_prob - min_prob, 0.0, prob_range) / prob_range
    size_val  = sz_base + sw_excess * (sz_max - sz_base)

    out["signal_long"]          = sig_long
    out["signal_short"]         = sig_short
    out["exit_long"]            = exit_long
    out["exit_short"]           = exit_short
    out["position_size_long"]   = np.where(sig_long  == 1, size_val, 0.0)
    out["position_size_short"]  = np.where(sig_short == 1, size_val, 0.0)
    out["signal_source"]        = np.where((sig_long == 1) | (sig_short == 1), "range", "")

    return out
