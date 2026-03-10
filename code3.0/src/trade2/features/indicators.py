"""
features/indicators.py - Pure technical indicator functions. No config dependency.
"""

import numpy as np
import pandas as pd
import talib


def hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).
    Reduces lag while remaining smooth.
    """
    half   = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = talib.WMA(series.values.astype(float), timeperiod=half)
    wma_full = talib.WMA(series.values.astype(float), timeperiod=period)
    raw      = 2 * wma_half - wma_full
    result   = talib.WMA(raw, timeperiod=sqrt_p)
    return pd.Series(result, index=series.index, name=f"HMA_{period}")


def compute_atr_pandas(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """
    ATR using pandas EWM (no TA-Lib dependency).
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
