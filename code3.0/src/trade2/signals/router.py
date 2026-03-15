"""
signals/router.py - Strategy router: classifies bars by regime and delegates
to specialized sub-strategies (trend, range, volatile).

Usage (multi-TF mode - regime already in df):
    df = forward_fill_1h_regime(df_5m, ...)
    signals = route_signals(df, config)

Usage (single-TF mode):
    signals = route_signals(df, config,
                            hmm_labels=labels, hmm_bull_prob=bull,
                            hmm_bear_prob=bear, hmm_index=idx)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from trade2.signals.generator import _session_mask, _is_5m_data
from trade2.signals.strategies.trend    import trend_strategy
from trade2.signals.strategies.range    import range_strategy
from trade2.signals.strategies.volatile import volatile_strategy
from trade2.signals.strategies.cdc      import cdc_strategy


def _empty_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with all signal columns zeroed out."""
    out = df.copy()
    for col in ("signal_long", "signal_short", "exit_long", "exit_short"):
        out[col] = 0
    for col in ("position_size_long", "position_size_short"):
        out[col] = 0.0
    out["signal_source"] = ""
    return out


def route_signals(
    df: pd.DataFrame,
    config: Dict[str, Any],
    # Single-TF mode: pass these explicitly
    hmm_labels: pd.Series = None,
    hmm_bull_prob: np.ndarray = None,
    hmm_bear_prob: np.ndarray = None,
    hmm_index: pd.Index = None,
) -> pd.DataFrame:
    """
    Classify each bar by regime probability and delegate to the appropriate
    sub-strategy. Masks are mutually exclusive (priority: trend > range > volatile > cdc).

    Returns df with signal_long, signal_short, exit_long, exit_short,
    position_size_long, position_size_short, signal_source columns.
    """
    strat_cfg    = config["strategies"]
    trend_cfg    = strat_cfg["trend"]
    range_cfg    = strat_cfg["range"]
    volatile_cfg = strat_cfg["volatile"]
    cdc_cfg      = strat_cfg["cdc"]
    reg_cfg      = config["regime"]
    hmm_cfg      = config["hmm"]

    out = df.copy()

    # ---- 1. Align regime columns ----
    if "regime" not in out.columns:
        if hmm_labels is None:
            raise ValueError("Either pass regime in df (multi_tf) or provide hmm_labels (single_tf)")
        regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
        bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
        bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")
        out["regime"]    = regime_s.reindex(out.index).fillna("sideways")
        out["bull_prob"] = bull_prob_s.reindex(out.index).fillna(0.0)
        out["bear_prob"] = bear_prob_s.reindex(out.index).fillna(0.0)

    if "sideways_prob" not in out.columns:
        out["sideways_prob"] = (1.0 - out["bull_prob"] - out["bear_prob"]).clip(lower=0.0)

    # ---- 2. Apply global transition cooldown ----
    cooldown = reg_cfg["transition_cooldown_bars"]
    if cooldown > 0:
        freq_mult     = 12 if _is_5m_data(out) else 1
        cooldown_bars = cooldown * freq_mult
        regime_changed = out["regime"] != out["regime"].shift(1)
        in_cooldown    = regime_changed.rolling(cooldown_bars, min_periods=1).sum() > 0
    else:
        in_cooldown = pd.Series(False, index=out.index)

    # ---- 3. Classify bars (priority: trend > range > volatile) ----
    bull_raw  = (out["bull_prob"]     >= trend_cfg["min_prob"])    & ~in_cooldown
    bear_raw  = (out["bear_prob"]     >= trend_cfg["min_prob"])    & ~in_cooldown
    range_raw = (out["sideways_prob"] >= range_cfg["min_prob"])    & ~in_cooldown

    max_prob = out[["bull_prob", "bear_prob", "sideways_prob"]].max(axis=1)
    vol_raw  = (max_prob < volatile_cfg["max_confidence"]) & ~in_cooldown

    trend_long  = bull_raw
    trend_short = bear_raw
    is_trend    = trend_long | trend_short
    range_mask  = range_raw & ~is_trend
    vol_mask    = vol_raw   & ~is_trend & ~range_mask

    # ---- 4. Call sub-strategies (with enabled guard) ----
    trend_enabled   = trend_cfg.get("enabled", True)
    range_enabled   = range_cfg.get("enabled", True)
    vol_enabled     = volatile_cfg.get("enabled", True)
    cdc_enabled     = cdc_cfg.get("enabled", False)

    trend_df   = trend_strategy(out,   config, trend_long, trend_short) if trend_enabled   else _empty_signals(out)
    range_df   = range_strategy(out,   config, range_mask)               if range_enabled   else _empty_signals(out)
    vol_df     = volatile_strategy(out, config, vol_mask)                 if vol_enabled     else _empty_signals(out)

    # CDC: regime_gated -> use trend masks; standalone -> all-True mask
    if cdc_enabled:
        if cdc_cfg["regime_gated"]:
            cdc_long_mask  = trend_long
            cdc_short_mask = trend_short
        else:
            cdc_long_mask  = pd.Series(True, index=out.index)
            cdc_short_mask = pd.Series(True, index=out.index)
        cdc_df = cdc_strategy(out, config, cdc_long_mask, cdc_short_mask)
    else:
        cdc_df = _empty_signals(out)

    # ---- 5. Merge signals ----
    result = out.copy()
    result["signal_long"]  = (
        (trend_df["signal_long"]  == 1) |
        (range_df["signal_long"]  == 1) |
        (vol_df["signal_long"]    == 1) |
        (cdc_df["signal_long"]    == 1)
    ).astype(int)

    result["signal_short"] = (
        (trend_df["signal_short"] == 1) |
        (range_df["signal_short"] == 1) |
        (vol_df["signal_short"]   == 1) |
        (cdc_df["signal_short"]   == 1)
    ).astype(int)

    result["exit_long"]  = (
        (trend_df["exit_long"]  == 1) &
        (range_df["exit_long"]  == 1) &
        (vol_df["exit_long"]    == 1) &
        (cdc_df["exit_long"]    == 1)
    ).astype(int)

    result["exit_short"] = (
        (trend_df["exit_short"] == 1) &
        (range_df["exit_short"] == 1) &
        (vol_df["exit_short"]   == 1) &
        (cdc_df["exit_short"]   == 1)
    ).astype(int)

    result["position_size_long"] = (
        trend_df["position_size_long"]  +
        range_df["position_size_long"]  +
        vol_df["position_size_long"]    +
        cdc_df["position_size_long"]
    )
    result["position_size_short"] = (
        trend_df["position_size_short"] +
        range_df["position_size_short"] +
        vol_df["position_size_short"]   +
        cdc_df["position_size_short"]
    )

    def _merge_source(t, r, v, c):
        out_arr = np.where(t != "", t, np.where(r != "", r, np.where(v != "", v, c)))
        return pd.Series(out_arr, index=result.index)

    result["signal_source"] = _merge_source(
        trend_df["signal_source"].values,
        range_df["signal_source"].values,
        vol_df["signal_source"].values,
        cdc_df["signal_source"].values,
    )

    # ---- 6. Diagnostics ----
    n_trend   = int((result["signal_source"] == "trend").sum())
    n_range   = int((result["signal_source"] == "range").sum())
    n_vol     = int((result["signal_source"] == "volatile").sum())
    n_cdc     = int((result["signal_source"] == "cdc").sum())
    n_total   = n_trend + n_range + n_vol + n_cdc
    print(f"  [router] signals: trend={n_trend} | range={n_range} | volatile={n_vol} | cdc={n_cdc} | total={n_total}")

    return result
