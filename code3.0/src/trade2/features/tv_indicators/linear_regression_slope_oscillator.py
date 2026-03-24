"""
features/linear_regression_slope_oscillator.py - Linear Regression Slope Oscillator.
Computes short-term linear regression slope momentum for scalping.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_linear_regression_slope_oscillator_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Linear Regression Slope Oscillator features for scalping.

    Uses short periods (3-14 bars) for 5M scalping charts.
    Bullish when slope > 0 (positive momentum), Bearish when slope < 0 (negative momentum).

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["linear_regression_slope_oscillator"].

    Returns:
        df copy with linear_regression_slope_oscillator columns added.
    """
    cfg = config.get('tv_indicators', {}).get('linear_regression_slope_oscillator', {})
    period = cfg.get('period', 8)

    # Scalping constraint: clamp period to 3-14 bars
    period = max(3, min(14, int(period)))

    out = df.copy()
    close = out["Close"].astype(float).values

    # Compute linear regression slope
    slope = talib.LINEARREG_SLOPE(close, timeperiod=period)

    # Determine bullish/bearish states
    bull_raw = slope > 0
    bear_raw = slope < 0

    # Shift(1) for lag safety
    idx = out.index
    out["linear_regression_slope_oscillator_slope"] = pd.Series(slope, index=idx).shift(1)
    out["linear_regression_slope_oscillator_bull"] = pd.Series(bull_raw, index=idx, dtype=bool).shift(1)
    out["linear_regression_slope_oscillator_bear"] = pd.Series(bear_raw, index=idx, dtype=bool).shift(1)

    return out