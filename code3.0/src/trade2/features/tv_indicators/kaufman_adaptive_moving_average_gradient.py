"""
features/kama_gradient.py - Kaufman Adaptive Moving Average Gradient.
Computes KAMA and its gradient for short-term momentum on 5M charts.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_kaufman_adaptive_moving_average_gradient_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute KAMA Gradient momentum signals for 5M scalping.
    
    Logic:
    1. Calculate KAMA (Kaufman Adaptive Moving Average) with short period
    2. Compute gradient = (KAMA - KAMA[n]) where n = gradient_lookback
    3. Bullish when gradient > 0 (rising momentum)
    4. Bearish when gradient < 0 (falling momentum)
    
    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads from config["tv_indicators"]["kaufman_adaptive_moving_average_gradient"]
    
    Returns:
        df copy with KAMA Gradient columns added.
    """
    # Get parameters with scalping-appropriate defaults
    params = config.get('tv_indicators', {}).get('kaufman_adaptive_moving_average_gradient', {})
    kama_period = params.get('kama_period', 10)  # Default 10, within 3-14 range
    gradient_lookback = params.get('gradient_lookback', 3)  # Default 3, within 3-5 range
    
    # Ensure scalping constraints
    kama_period = min(max(kama_period, 3), 14)
    gradient_lookback = min(max(gradient_lookback, 2), 5)
    
    out = df.copy()
    close = out["Close"].astype(float).values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    kama = talib.KAMA(close, timeperiod=kama_period)
    
    # Calculate gradient (rate of change over lookback period)
    gradient = kama - np.roll(kama, gradient_lookback)
    
    # Create signals
    bull_signal = gradient > 0
    bear_signal = gradient < 0
    
    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr
    
    # Apply shift(1) for lag safety
    idx = out.index
    out["kaufman_adaptive_moving_average_gradient_kama"] = pd.Series(kama, index=idx).shift(1)
    out["kaufman_adaptive_moving_average_gradient_gradient"] = pd.Series(gradient, index=idx).shift(1)
    out["kaufman_adaptive_moving_average_gradient_bull"] = pd.Series(_lag1(bull_signal), index=idx, dtype=bool)
    out["kaufman_adaptive_moving_average_gradient_bear"] = pd.Series(_lag1(bear_signal), index=idx, dtype=bool)
    
    return out