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

    out["regime"]    = regime_s.reindex(out.index).ffill().fillna("sideways")
    out["bull_prob"] = bull_prob_s.reindex(out.index).ffill().fillna(0.0)
    out["bear_prob"] = bear_prob_s.reindex(out.index).ffill().fillna(0.0)

    if hmm_sideways_prob is not None:
        sideways_s = pd.Series(hmm_sideways_prob, index=hmm_index, name="sideways_prob")
        out["sideways_prob"] = sideways_s.reindex(out.index).ffill().fillna(0.0)

    if atr_1h is not None:
        out["atr_1h"] = atr_1h.reindex(out.index).ffill().fillna(atr_1h.median())

    if hma_rising is not None:
        out["hma_rising"] = hma_rising.reindex(out.index).ffill().fillna(False).astype(bool)

    if price_above_hma is not None:
        out["price_above_hma"] = price_above_hma.reindex(out.index).ffill().fillna(False).astype(bool)

    return out
