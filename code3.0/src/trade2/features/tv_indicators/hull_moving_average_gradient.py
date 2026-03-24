"""
features/hull_moving_average_gradient.py - Hull MA Gradient momentum detection.
Computes HMA gradient (rate of change) for acceleration signals.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_hull_moving_average_gradient_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Hull Moving Average Gradient momentum features.

    Signals:
      hull_moving_average_gradient_bull: gradient > 0 (positive momentum)
      hull_moving_average_gradient_bear: gradient < 0 (negative momentum)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["hull_moving_average_gradient"].

    Returns:
        df copy with HMA Gradient columns added.
    """
    cfg = config.get('tv_indicators', {}).get('hull_moving_average_gradient', {})
    hma_period = cfg.get("hma_period", 8)  # SCALPING: default to 8 for 5M
    gradient_lookback = cfg.get("gradient_lookback", 3)  # SCALPING: default 3
    
    # Enforce scalping constraints
    hma_period = max(3, min(14, hma_period))
    gradient_lookback = max(2, min(10, gradient_lookback))
    
    out = df.copy()
    close = out["Close"].astype(float).values
    
    # Hull Moving Average calculation
    # HMA = WMA(2*WMA(period/2) - WMA(period), sqrt(period))
    half_period = int(np.round(hma_period / 2))
    sqrt_period = int(np.round(np.sqrt(hma_period)))
    
    wma_half = talib.WMA(close, timeperiod=half_period)
    wma_full = talib.WMA(close, timeperiod=hma_period)
    diff = 2 * wma_half - wma_full
    hma = talib.WMA(diff, timeperiod=sqrt_period)
    
    # Gradient (rate of change) calculation
    # gradient = (hma - hma[n]) / n where n = gradient_lookback
    hma_series = pd.Series(hma, index=out.index)
    gradient = (hma_series - hma_series.shift(gradient_lookback)) / gradient_lookback
    
    # Bullish/bearish signals
    bull_raw = gradient > 0
    bear_raw = gradient < 0
    
    # Shift(1) for lag safety
    out["hull_moving_average_gradient_hma"] = hma_series.shift(1)
    out["hull_moving_average_gradient_gradient"] = gradient.shift(1)
    out["hull_moving_average_gradient_bull"] = bull_raw.shift(1).astype(bool)
    out["hull_moving_average_gradient_bear"] = bear_raw.shift(1).astype(bool)
    
    return out