"""
signals/regime.py - Forward-fill 1H HMM regime onto 5M bars.
"""

import numpy as np
import pandas as pd


def forward_fill_1h_regime(
    df_5m: pd.DataFrame,
    hmm_labels: pd.Series,
    hmm_bull_prob: np.ndarray,
    hmm_bear_prob: np.ndarray,
    hmm_index: pd.Index,
    atr_1h: pd.Series = None,
    hma_rising: pd.Series = None,
    price_above_hma: pd.Series = None,
    hmm_sideways_prob: np.ndarray = None,
) -> pd.DataFrame:
    """
    Forward-fill 1H HMM regime labels and probabilities onto 5M bars.

    Each 5M bar gets the regime from the most recently completed 1H bar.
    Lag-safe: 1H bar must be fully closed before its regime is used.

    Args:
        df_5m:         5M feature DataFrame
        hmm_labels:    Series of regime labels (indexed 0..N by position)
        hmm_bull_prob: Array of bull posterior probabilities
        hmm_bear_prob: Array of bear posterior probabilities
        hmm_index:     DatetimeIndex corresponding to HMM predictions (1H bar opens)
        atr_1h:        Optional 1H ATR series (indexed by 1H timestamps).
                       If provided, forward-filled onto 5M bars as 'atr_1h'.
                       Used by compute_stops for appropriately-sized SL/TP in multi-TF mode.

    Returns:
        df_5m copy with regime, bull_prob, bear_prob columns added.
        If atr_1h provided, also adds atr_1h column.
    """
    out = df_5m.copy()

    regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
    bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
    bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")

    # Use method="ffill" so that each 5M bar gets the regime from the most
    # recently completed 1H bar even when no 1H timestamp falls exactly on a
    # 5M boundary (which is always the case in live MT5 where the two windows
    # may not share any exact timestamps).
    out["regime"]    = regime_s.reindex(out.index, method="ffill").fillna("sideways")
    out["bull_prob"] = bull_prob_s.reindex(out.index, method="ffill").fillna(0.0)
    out["bear_prob"] = bear_prob_s.reindex(out.index, method="ffill").fillna(0.0)

    if hmm_sideways_prob is not None:
        sideways_s = pd.Series(hmm_sideways_prob, index=hmm_index, name="sideways_prob")
        out["sideways_prob"] = sideways_s.reindex(out.index, method="ffill").fillna(0.0)

    if atr_1h is not None:
        out["atr_1h"] = atr_1h.reindex(out.index, method="ffill").fillna(atr_1h.median())

    if hma_rising is not None:
        out["hma_rising"] = hma_rising.reindex(out.index, method="ffill").fillna(False).astype(bool)

    if price_above_hma is not None:
        out["price_above_hma"] = price_above_hma.reindex(out.index, method="ffill").fillna(False).astype(bool)

    return out


def forward_fill_cdc_levels(
    df_signal: pd.DataFrame,
    df_regime_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill CDC Action Zone levels from regime TF onto signal TF bars.

    Picks up cdc_fast, cdc_slow, cdc_zone_green, cdc_zone_red from df_regime_feat
    and forward-fills them onto df_signal with a _15m suffix.
    Also computes cdc_bullish_15m (Fast > Slow).

    Args:
        df_signal:      Signal TF DataFrame (e.g. 1M bars).
        df_regime_feat: Regime TF feature DataFrame with CDC columns.

    Returns:
        df_signal copy with cdc_fast_15m, cdc_slow_15m, cdc_bullish_15m,
        cdc_zone_green_15m, cdc_zone_red_15m columns added.
    """
    out = df_signal.copy()

    cdc_cols = {
        "cdc_fast":       "cdc_fast_15m",
        "cdc_slow":       "cdc_slow_15m",
        "cdc_zone_green": "cdc_zone_green_15m",
        "cdc_zone_red":   "cdc_zone_red_15m",
    }

    for src_col, dst_col in cdc_cols.items():
        if src_col not in df_regime_feat.columns:
            continue
        s = df_regime_feat[src_col]
        out[dst_col] = s.reindex(out.index, method="ffill").fillna(
            False if df_regime_feat[src_col].dtype == bool else 0.0
        )

    # Derived: bullish when Fast > Slow on regime TF
    if "cdc_fast_15m" in out.columns and "cdc_slow_15m" in out.columns:
        out["cdc_bullish_15m"] = out["cdc_fast_15m"] > out["cdc_slow_15m"]

    return out


def forward_fill_5m_to_1m(
    df_1m: pd.DataFrame,
    df_5m_signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill 5M signal columns onto 1M bars for the PullbackEngine.

    Columns forwarded from df_5m_signals:
      signal_long, signal_short, exit_long, exit_short
      stop_long, stop_short, tp_long, tp_short
      position_size_long, position_size_short
      atr_14   (forwarded as atr_5m)
      Close    (forwarded as signal_close_5m  -- 5M bar close for tier calc)
      Volume   (forwarded as vol_5m)
      regime, bull_prob, bear_prob

    Also adds:
      vol_avg_5m   -- 20-bar rolling mean of vol_5m (for vol_ratio)
      new_signal_long  -- 1 on first 1M bar where signal_long_5m transitions 0->1
      new_signal_short -- 1 on first 1M bar where signal_short_5m transitions 0->1

    Args:
        df_1m:          1M OHLCV DataFrame with DatetimeIndex
        df_5m_signals:  5M DataFrame after generate_signals + compute_stops

    Returns:
        df_1m copy with all forwarded columns added.
    """
    out = df_1m.copy()

    # Columns to forward-fill directly (src_col -> dst_col)
    fwd_cols = {
        "signal_long":           "signal_long_5m",
        "signal_short":          "signal_short_5m",
        "exit_long":             "exit_long_5m",
        "exit_short":            "exit_short_5m",
        "stop_long":             "stop_long_5m",
        "stop_short":            "stop_short_5m",
        "tp_long":               "tp_long_5m",
        "tp_short":              "tp_short_5m",
        "position_size_long":    "position_size_long_5m",
        "position_size_short":   "position_size_short_5m",
        "atr_14":                "atr_5m",
        "Close":                 "signal_close_5m",
        "Volume":                "vol_5m",
        "regime":                "regime",
        "bull_prob":             "bull_prob",
        "bear_prob":             "bear_prob",
    }

    for src_col, dst_col in fwd_cols.items():
        if src_col not in df_5m_signals.columns:
            continue
        s = df_5m_signals[src_col]
        dtype = df_5m_signals[src_col].dtype
        fill_val = 0 if dtype in [int, bool, "int64", "bool"] or str(dtype) in ["int64", "bool"] else 0.0
        out[dst_col] = s.reindex(out.index, method="ffill").fillna(fill_val)

    # 20-bar rolling mean of Volume on 5M (for vol_ratio in pullback engine)
    if "Volume" in df_5m_signals.columns:
        vol_avg = df_5m_signals["Volume"].rolling(20, min_periods=1).mean()
        out["vol_avg_5m"] = vol_avg.reindex(out.index, method="ffill").fillna(
            float(df_5m_signals["Volume"].mean())
        )

    # Onset markers: 1 on first 1M bar where signal transitions 0->1
    for sig_col, onset_col in [
        ("signal_long_5m",  "new_signal_long"),
        ("signal_short_5m", "new_signal_short"),
    ]:
        if sig_col in out.columns:
            prev = out[sig_col].shift(1).fillna(0)
            out[onset_col] = ((out[sig_col] == 1) & (prev == 0)).astype(int)
        else:
            out[onset_col] = 0

    return out
