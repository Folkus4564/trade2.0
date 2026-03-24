"""
features/rsi_extreme_snapback.py - RSI Extreme Snapback feature detection.
Detects when RSI reaches a short-term extreme and snaps back toward the middle.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_rsi_extreme_snapback_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute RSI Extreme Snapback features.

    Logic:
      - Compute short-period RSI (default 5)
      - Bull snapback: RSI was below lower_band (default 20) on previous bar,
        and has now crossed back above it (snapback from oversold extreme)
      - Bear snapback: RSI was above upper_band (default 80) on previous bar,
        and has now crossed back below it (snapback from overbought extreme)

    Additional confirmation columns:
      - rsi_extreme_snapback_rsi:        Raw RSI value (shifted)
      - rsi_extreme_snapback_oversold:   RSI < lower_band (shifted)
      - rsi_extreme_snapback_overbought: RSI > upper_band (shifted)
      - rsi_extreme_snapback_bull:       Bullish snapback signal (shifted)
      - rsi_extreme_snapback_bear:       Bearish snapback signal (shifted)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config['tv_indicators']['rsi_extreme_snapback'].

    Returns:
        df copy with rsi_extreme_snapback_ columns added.
    """
    params = config.get('tv_indicators', {}).get('rsi_extreme_snapback', {})
    rsi_period  = int(params.get('rsi_period',  5))
    lower_band  = float(params.get('lower_band', 20.0))
    upper_band  = float(params.get('upper_band', 80.0))

    # Clamp periods to scalping-safe range
    rsi_period = max(3, min(rsi_period, 14))

    out   = df.copy()
    close = out['Close'].astype(float).values

    # --- RSI ---
    rsi = talib.RSI(close, timeperiod=rsi_period)

    # --- Zone flags (raw, pre-shift) ---
    oversold   = rsi < lower_band   # RSI in oversold territory
    overbought = rsi > upper_band   # RSI in overbought territory

    # --- Helper: 1-bar numpy lag ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        lagged = np.empty_like(arr)
        lagged[0] = False
        lagged[1:] = arr[:-1]
        return lagged

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        lagged = np.empty_like(arr)
        lagged[0] = np.nan
        lagged[1:] = arr[:-1]
        return lagged

    # --- Snapback signals (raw, pre-shift) ---
    # Bull: was oversold last bar, now crossed back above lower_band
    bull_raw = _lag1(oversold) & (rsi >= lower_band)

    # Bear: was overbought last bar, now crossed back below upper_band
    bear_raw = _lag1(overbought) & (rsi <= upper_band)

    # --- Assign outputs with shift(1) for lag safety ---
    idx = out.index

    out['rsi_extreme_snapback_rsi']        = pd.Series(_lag1_float(rsi),        index=idx, dtype=float)
    out['rsi_extreme_snapback_oversold']   = pd.Series(_lag1(oversold),         index=idx, dtype=bool)
    out['rsi_extreme_snapback_overbought'] = pd.Series(_lag1(overbought),       index=idx, dtype=bool)
    out['rsi_extreme_snapback_bull']       = pd.Series(_lag1(bull_raw),         index=idx, dtype=bool)
    out['rsi_extreme_snapback_bear']       = pd.Series(_lag1(bear_raw),         index=idx, dtype=bool)

    return out