"""
features/coppock_curve.py - Coppock Curve momentum oscillator.
Computes long-term momentum for identifying major market bottoms.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_coppock_curve_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Coppock Curve features.

    Formula:
        ROC1 = 10-period rate of change
        ROC2 = 14-period rate of change
        Coppock = WMA(10) of (ROC1 + ROC2)

    Signals:
        coppock_curve_bull: True when indicator value > 0
        coppock_curve_bear: True when indicator value < 0

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["coppock_curve"].

    Returns:
        df copy with Coppock Curve columns added.
    """
    cfg = config.get('tv_indicators', {}).get('coppock_curve', {})
    wma_period = cfg.get('wma_period', 10)
    roc_long   = cfg.get('roc_long', 14)
    roc_short  = cfg.get('roc_short', 11)

    out = df.copy()
    close = out["Close"].astype(float).values

    # Calculate Rate of Change (ROC)
    roc1 = talib.ROC(close, timeperiod=roc_short)
    roc2 = talib.ROC(close, timeperiod=roc_long)

    # Sum of ROCs and apply Weighted Moving Average
    roc_sum = roc1 + roc2
    coppock = talib.WMA(roc_sum, timeperiod=wma_period)

    # Generate raw signals
    bull_raw = coppock > 0
    bear_raw = coppock < 0

    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    # Apply lag safety (shift(1))
    idx = out.index
    out["coppock_curve_value"] = pd.Series(coppock, index=idx).shift(1)
    out["coppock_curve_bull"]  = pd.Series(_lag1(bull_raw), index=idx, dtype=bool)
    out["coppock_curve_bear"]  = pd.Series(_lag1(bear_raw), index=idx, dtype=bool)

    return out