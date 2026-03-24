"""
features/fast_macd_histogram.py - Fast MACD Histogram momentum signals.
Uses short periods (fast=5, slow=13, signal=4) for rapid momentum detection.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_fast_macd_histogram_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute fast MACD histogram signals for 5-minute scalping.

    Signals:
      fast_macd_histogram_bull: Histogram > 0 (positive momentum)
      fast_macd_histogram_bear: Histogram < 0 (negative momentum)

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["fast_macd_histogram"].

    Returns:
        df copy with fast_macd_histogram columns added.
    """
    # Get parameters with defaults
    params = config.get("tv_indicators", {}).get("fast_macd_histogram", {})
    fast_length = params.get("fast_length", 5)
    slow_length = params.get("slow_length", 13)
    signal_length = params.get("signal_length", 4)

    # Validate scalping constraints
    fast_length = min(max(3, fast_length), 14)
    slow_length = min(max(3, slow_length), 14)
    signal_length = min(max(3, signal_length), 14)

    out = df.copy()
    close = out["Close"].astype(float).values

    # Calculate fast MACD
    macd, signal, histogram = talib.MACD(
        close,
        fastperiod=fast_length,
        slowperiod=slow_length,
        signalperiod=signal_length
    )

    # Generate raw signals
    bull_raw = histogram > 0
    bear_raw = histogram < 0

    # Apply shift(1) for lag safety using numpy method
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr, dtype=bool if arr.dtype == bool else float)
        out_arr[0] = False if arr.dtype == bool else 0.0
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index
    out["fast_macd_histogram_hist"] = pd.Series(_lag1(histogram), index=idx, dtype=float)
    out["fast_macd_histogram_macd"] = pd.Series(_lag1(macd), index=idx, dtype=float)
    out["fast_macd_histogram_signal"] = pd.Series(_lag1(signal), index=idx, dtype=float)
    out["fast_macd_histogram_bull"] = pd.Series(_lag1(bull_raw), index=idx, dtype=bool)
    out["fast_macd_histogram_bear"] = pd.Series(_lag1(bear_raw), index=idx, dtype=bool)

    return out