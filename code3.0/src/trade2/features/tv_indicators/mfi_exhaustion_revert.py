"""
features/mfi_exhaustion_revert.py - MFI Exhaustion Revert feature detection.
Detects Money Flow Index reaching extremes and price reverting from range edges.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_mfi_exhaustion_revert_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute MFI Exhaustion Revert features.

    Detects when the Money Flow Index reaches an extreme (overbought/oversold)
    and price begins to revert from a recent range edge.

    Signals:
      mfi_exhaustion_revert_bull: MFI was oversold (below lower_band) and is now
                                   rising back above it, with price near/at range low
      mfi_exhaustion_revert_bear: MFI was overbought (above upper_band) and is now
                                   falling back below it, with price near/at range high

    Additional columns:
      mfi_exhaustion_revert_mfi:        raw MFI value (shifted)
      mfi_exhaustion_revert_range_high: recent range high (shifted)
      mfi_exhaustion_revert_range_low:  recent range low (shifted)
      mfi_exhaustion_revert_oversold:   MFI below lower_band (shifted)
      mfi_exhaustion_revert_overbought: MFI above upper_band (shifted)

    Args:
        df:     OHLCV DataFrame with Open, High, Low, Close, Volume columns
        config: Full config dict. Reads config["tv_indicators"]["mfi_exhaustion_revert"].

    Returns:
        df copy with mfi_exhaustion_revert_ columns added.
    """
    params = config.get("tv_indicators", {}).get("mfi_exhaustion_revert", {})

    lower_band = int(params.get("lower_band", 15))
    upper_band = int(params.get("upper_band", 85))
    mfi_period = int(params.get("mfi_period", 7))

    # Clamp mfi_period to scalping-safe range
    mfi_period = max(3, min(mfi_period, 14))

    # Range lookback for detecting price at range edge (short, scalping-suitable)
    range_lookback = min(max(mfi_period + 2, 5), 12)

    out = df.copy()
    high   = out["High"].astype(float).values
    low    = out["Low"].astype(float).values
    close  = out["Close"].astype(float).values
    volume = out["Volume"].astype(float).values

    # --- MFI via talib ---
    mfi = talib.MFI(high, low, close, volume, timeperiod=mfi_period)

    # --- Recent range high/low over range_lookback bars ---
    n = len(close)
    range_high = np.full(n, np.nan)
    range_low  = np.full(n, np.nan)

    for i in range(range_lookback - 1, n):
        window_high = high[i - range_lookback + 1: i + 1]
        window_low  = low[i  - range_lookback + 1: i + 1]
        range_high[i] = np.max(window_high)
        range_low[i]  = np.min(window_low)

    # --- Zone flags (raw, before shift) ---
    oversold   = mfi < lower_band
    overbought = mfi > upper_band

    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = np.nan
        out_arr[1:] = arr[:-1]
        return out_arr

    # --- Revert signals ---
    # Bull: MFI crosses back above lower_band (was oversold, now recovering)
    #       AND close is within the lower portion of the recent range (price at range low edge)
    range_span  = range_high - range_low
    lower_third = range_low + range_span * 0.35

    # MFI crosses above lower_band: previous bar oversold, current bar not oversold
    mfi_cross_up = ~oversold & _lag1(oversold)

    # Price near range low: close below the lower 35% threshold of range
    price_at_low = close <= lower_third

    bull_raw = mfi_cross_up & price_at_low

    # Bear: MFI crosses back below upper_band (was overbought, now fading)
    #       AND close is within the upper portion of the recent range (price at range high edge)
    upper_third = range_high - range_span * 0.35

    mfi_cross_down = ~overbought & _lag1(overbought)

    price_at_high = close >= upper_third

    bear_raw = mfi_cross_down & price_at_high

    # --- Apply shift(1) for lag safety ---
    idx = out.index

    out["mfi_exhaustion_revert_mfi"]        = pd.Series(_lag1_float(mfi),        index=idx, dtype=float)
    out["mfi_exhaustion_revert_range_high"] = pd.Series(_lag1_float(range_high), index=idx, dtype=float)
    out["mfi_exhaustion_revert_range_low"]  = pd.Series(_lag1_float(range_low),  index=idx, dtype=float)
    out["mfi_exhaustion_revert_oversold"]   = pd.Series(_lag1(oversold),         index=idx, dtype=bool)
    out["mfi_exhaustion_revert_overbought"] = pd.Series(_lag1(overbought),       index=idx, dtype=bool)
    out["mfi_exhaustion_revert_bull"]       = pd.Series(_lag1(bull_raw),         index=idx, dtype=bool)
    out["mfi_exhaustion_revert_bear"]       = pd.Series(_lag1(bear_raw),         index=idx, dtype=bool)

    return out