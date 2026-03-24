"""
features/eckhardt_impulse_oscillator.py - Eckhardt Impulse Oscillator feature detection.
A zero-lag momentum oscillator using double-smoothed EMA for fast scalping signals.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_eckhardt_impulse_oscillator_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Eckhardt Impulse Oscillator features for 5M scalping.
    
    Signal logic:
      bullish = impulse > 0 AND impulse > impulse_ema (positive momentum)
      bearish = impulse < 0 AND impulse < impulse_ema (negative momentum)
    
    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["eckhardt_impulse_oscillator"].
    
    Returns:
        df copy with EIO columns added.
    """
    indicator_cfg = config.get('tv_indicators', {}).get('eckhardt_impulse_oscillator', {})
    ema_length = indicator_cfg.get('ema_length', 8)  # Short for scalping
    impulse_length = indicator_cfg.get('impulse_length', 5)  # Short for scalping
    signal_period = 3  # Fixed short period for scalping
    
    # Cap periods for 5M scalping
    ema_length = min(max(ema_length, 3), 14)
    impulse_length = min(max(impulse_length, 3), 10)
    
    out = df.copy()
    close = out["Close"].astype(float).values
    
    # Zero-lag EMA calculation
    ema1 = talib.EMA(close, timeperiod=ema_length)
    ema2 = talib.EMA(ema1, timeperiod=ema_length)
    zero_lag_ema = 2 * ema1 - ema2
    
    # Impulse oscillator
    impulse = zero_lag_ema - np.roll(zero_lag_ema, impulse_length)
    impulse[:impulse_length] = 0
    
    # Signal line (EMA of impulse)
    impulse_ema = talib.EMA(impulse, timeperiod=signal_period)
    
    # Raw signals
    bull_raw = (impulse > 0) & (impulse > impulse_ema)
    bear_raw = (impulse < 0) & (impulse < impulse_ema)
    
    # Apply 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr
    
    idx = out.index
    out["eckhardt_impulse_oscillator_impulse"] = pd.Series(impulse, index=idx).shift(1)
    out["eckhardt_impulse_oscillator_impulse_ema"] = pd.Series(impulse_ema, index=idx).shift(1)
    out["eckhardt_impulse_oscillator_bull"] = pd.Series(_lag1(bull_raw), index=idx, dtype=bool)
    out["eckhardt_impulse_oscillator_bear"] = pd.Series(_lag1(bear_raw), index=idx, dtype=bool)
    
    return out