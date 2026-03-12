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
    sub-strategy. Masks are mutually exclusive (priority: trend > range > volatile).

    Returns df with signal_long, signal_short, exit_long, exit_short,
    position_size_long, position_size_short, signal_source columns.
    """
    strat_cfg   = config["strategies"]
    trend_cfg   = strat_cfg["trend"]
    range_cfg   = strat_cfg["range"]
    volatile_cfg = strat_cfg["volatile"]
    reg_cfg     = config["regime"]
    hmm_cfg     = config["hmm"]

    out = df.copy()

    # ---- 1. Align regime columns (same as generate_signals) ----
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
    bull_raw  = (out["bull_prob"]     >= trend_cfg["min_prob"])   & ~in_cooldown
    bear_raw  = (out["bear_prob"]     >= trend_cfg["min_prob"])   & ~in_cooldown
    range_raw = (out["sideways_prob"] >= range_cfg["min_prob"])   & ~in_cooldown

    max_prob = out[["bull_prob", "bear_prob", "sideways_prob"]].max(axis=1)
    vol_raw  = (max_prob < volatile_cfg["max_confidence"]) & ~in_cooldown

    # Enforce exclusivity: trend first, then range, then volatile
    trend_long  = bull_raw
    trend_short = bear_raw
    is_trend    = trend_long | trend_short

    range_mask  = range_raw & ~is_trend
    vol_mask    = vol_raw   & ~is_trend & ~range_mask

    # ---- 4. Call sub-strategies ----
    trend_df   = trend_strategy(out,   config, trend_long, trend_short)
    range_df   = range_strategy(out,   config, range_mask)
    vol_df     = volatile_strategy(out, config, vol_mask)

    # ---- 5. Merge (masks are exclusive, no conflicts) ----
    # signal columns: OR across sub-strategies
    result = out.copy()
    result["signal_long"]  = (
        (trend_df["signal_long"]  == 1) |
        (range_df["signal_long"]  == 1) |
        (vol_df["signal_long"]    == 1)
    ).astype(int)

    result["signal_short"] = (
        (trend_df["signal_short"] == 1) |
        (range_df["signal_short"] == 1) |
        (vol_df["signal_short"]   == 1)
    ).astype(int)

    # Exit: AND (position exits when ANY sub-strategy would exit)
    result["exit_long"]  = (
        (trend_df["exit_long"]  == 1) &
        (range_df["exit_long"]  == 1) &
        (vol_df["exit_long"]    == 1)
    ).astype(int)

    result["exit_short"] = (
        (trend_df["exit_short"] == 1) &
        (range_df["exit_short"] == 1) &
        (vol_df["exit_short"]   == 1)
    ).astype(int)

    # Position size: take the non-zero value from the active sub-strategy
    result["position_size_long"] = (
        trend_df["position_size_long"]  +
        range_df["position_size_long"]  +
        vol_df["position_size_long"]
    )
    result["position_size_short"] = (
        trend_df["position_size_short"] +
        range_df["position_size_short"] +
        vol_df["position_size_short"]
    )

    # Signal source: combine tags (one non-empty per bar due to exclusivity)
    def _merge_source(t, r, v):
        out_arr = np.where(t != "", t, np.where(r != "", r, v))
        return pd.Series(out_arr, index=result.index)

    result["signal_source"] = _merge_source(
        trend_df["signal_source"].values,
        range_df["signal_source"].values,
        vol_df["signal_source"].values,
    )

    # ---- 6. Diagnostics ----
    n_trend   = int((result["signal_source"] == "trend").sum())
    n_range   = int((result["signal_source"] == "range").sum())
    n_vol     = int((result["signal_source"] == "volatile").sum())
    n_total   = n_trend + n_range + n_vol
    print(f"  [router] signals: trend={n_trend} | range={n_range} | volatile={n_vol} | total={n_total}")

    return result
