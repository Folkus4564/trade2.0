"""
features/williams_r.py - Williams %R oscillator feature detection.
Computes overbought/oversold zone transitions.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_williams_r_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Williams %R oscillator features.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    Ranges from 0 to -100 (negative scale).

    Zones:
      Oversold (bullish potential): %R < -80
      Overbought (bearish potential): %R > -20

    Signals:
      williams_r_bull: First bar entering oversold zone (shift(1))
      williams_r_bear: First bar entering overbought zone (shift(1))

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["williams_r"].

    Returns:
        df copy with Williams %R columns added.
    """
    will_cfg = config.get("tv_indicators", {}).get("williams_r", {})
    period = will_cfg.get("period", 14)

    out = df.copy()
    high = out["High"].astype(float).values
    low = out["Low"].astype(float).values
    close = out["Close"].astype(float).values

    # Highest high and lowest low over period
    hh = talib.MAX(high, timeperiod=period)
    ll = talib.MIN(low, timeperiod=period)

    # Williams %R calculation (handle division by zero)
    numerator = hh - close
    denominator = hh - ll
    willr = np.full_like(close, 0.0)  # Default 0 when denominator 0
    mask = denominator != 0
    willr[mask] = (numerator[mask] / denominator[mask]) * -100.0

    # Zone classification (raw, before shift)
    oversold_zone = willr < -80
    overbought_zone = willr > -20

    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    # Transition signals: entering oversold or overbought zones
    will_bull_raw = oversold_zone & ~_lag1(oversold_zone)
    will_bear_raw = overbought_zone & ~_lag1(overbought_zone)

    # Shift(1) for lag safety
    idx = out.index
    out["williams_r"] = pd.Series(willr, index=idx).shift(1)
    out["williams_r_bull"] = pd.Series(_lag1(will_bull_raw), index=idx, dtype=bool)
    out["williams_r_bear"] = pd.Series(_lag1(will_bear_raw), index=idx, dtype=bool)

    return out