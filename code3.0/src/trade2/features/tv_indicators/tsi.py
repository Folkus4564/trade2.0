"""
features/tsi.py - True Strength Index feature detection.
Computes double-smoothed momentum oscillator with signal line.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_tsi_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute True Strength Index features.

    Logic:
      1. Calculate price change: diff = Close - prev(Close)
      2. First smoothing: ema1 = EMA(diff, short_period)
      3. Second smoothing: ema2 = EMA(ema1, long_period)
      4. Absolute smoothing: abs_ema1 = EMA(|diff|, short_period)
      5. Second abs smoothing: abs_ema2 = EMA(abs_ema1, long_period)
      6. TSI = 100 * (ema2 / abs_ema2)
      7. Signal = EMA(TSI, signal_period)
      8. Bullish when TSI > Signal line
      9. Bearish when TSI < Signal line

    Args:
        df: OHLCV DataFrame with 'Close' column
        config: Full config dict. Reads config["tv_indicators"]["tsi"]

    Returns:
        df copy with TSI columns added, all shift(1)
    """
    tsi_cfg = config.get('tv_indicators', {}).get('tsi', {})
    long_period = tsi_cfg.get("long_period", 25)
    short_period = tsi_cfg.get("short_period", 13)
    signal_period = tsi_cfg.get("signal", 7)

    out = df.copy()
    close = out["Close"].astype(float).values

    # Calculate momentum
    momentum = np.zeros_like(close)
    momentum[1:] = close[1:] - close[:-1]
    momentum[0] = momentum[1]  # Handle first value

    # First EMA smoothing
    ema1 = talib.EMA(momentum, timeperiod=short_period)
    abs_ema1 = talib.EMA(np.abs(momentum), timeperiod=short_period)

    # Second EMA smoothing
    ema2 = talib.EMA(ema1, timeperiod=long_period)
    abs_ema2 = talib.EMA(abs_ema1, timeperiod=long_period)

    # Calculate TSI
    tsi = np.zeros_like(close)
    mask = abs_ema2 != 0
    tsi[mask] = 100 * (ema2[mask] / abs_ema2[mask])

    # Signal line
    signal = talib.EMA(tsi, timeperiod=signal_period)

    # Bullish/bearish conditions
    tsi_bull_raw = tsi > signal
    tsi_bear_raw = tsi < signal

    # Helper for 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    # Apply lag(1)
    idx = out.index
    out["tsi_value"] = pd.Series(tsi, index=idx).shift(1)
    out["tsi_signal"] = pd.Series(signal, index=idx).shift(1)
    out["tsi_bull"] = pd.Series(_lag1(tsi_bull_raw), index=idx, dtype=bool)
    out["tsi_bear"] = pd.Series(_lag1(tsi_bear_raw), index=idx, dtype=bool)

    return out