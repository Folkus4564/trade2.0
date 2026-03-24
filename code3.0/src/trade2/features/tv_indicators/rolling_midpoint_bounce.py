"""
features/rolling_midpoint_bounce.py - Rolling Midpoint Bounce feature detection.
Price mean-reverts from short rolling high-low midpoint deviations.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_rolling_midpoint_bounce_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Rolling Midpoint Bounce features.

    Logic:
      - Compute rolling midpoint = (rolling_high + rolling_low) / 2 over 'period' bars
      - Compute rolling range = rolling_high - rolling_low
      - Deviation = Close - midpoint
      - Normalized deviation = deviation / (range + epsilon)
      - Upper band = midpoint + deviation_mult * (range / 2)
      - Lower band = midpoint - deviation_mult * (range / 2)

    Signals:
      bull: price is below lower band (oversold vs midpoint) -> bounce up expected
      bear: price is above upper band (overbought vs midpoint) -> reversion down expected

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["rolling_midpoint_bounce"].

    Returns:
        df copy with rolling_midpoint_bounce_ columns added.
    """
    params = config.get("tv_indicators", {}).get("rolling_midpoint_bounce", {})
    period         = int(params.get("period", 8))
    deviation_mult = float(params.get("deviation_mult", 0.8))

    # Clamp period to scalping range
    period = max(3, min(period, 14))

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values

    # Rolling high and low using talib
    rolling_high = talib.MAX(high, timeperiod=period)
    rolling_low  = talib.MIN(low,  timeperiod=period)

    # Midpoint and range
    midpoint     = (rolling_high + rolling_low) / 2.0
    rolling_range = rolling_high - rolling_low

    epsilon = 1e-10

    # Half-range band offset
    half_range = rolling_range / 2.0

    upper_band = midpoint + deviation_mult * half_range
    lower_band = midpoint - deviation_mult * half_range

    # Deviation from midpoint normalized by range
    deviation_norm = (close - midpoint) / (rolling_range + epsilon)

    # Momentum: rate of change of normalized deviation using short EMA smoothing
    deviation_norm_series = pd.Series(deviation_norm)
    signal_period = max(3, min(period // 2, 5))
    smoothed_dev  = talib.EMA(deviation_norm, timeperiod=signal_period)

    # Raw signals (before shift)
    # Bull: close below lower band (oversold relative to midpoint) -> mean reversion bounce up
    bull_raw = close < lower_band

    # Bear: close above upper band (overbought relative to midpoint) -> mean reversion down
    bear_raw = close > upper_band

    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty(len(arr), dtype=bool)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty(len(arr), dtype=float)
        out_arr[0] = np.nan
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    # Boolean signals (shift 1)
    out["rolling_midpoint_bounce_bull"] = pd.Series(_lag1(bull_raw),  index=idx, dtype=bool)
    out["rolling_midpoint_bounce_bear"] = pd.Series(_lag1(bear_raw),  index=idx, dtype=bool)

    # Supporting columns (shift 1)
    out["rolling_midpoint_bounce_midpoint"]      = pd.Series(_lag1_float(midpoint),      index=idx)
    out["rolling_midpoint_bounce_upper_band"]    = pd.Series(_lag1_float(upper_band),    index=idx)
    out["rolling_midpoint_bounce_lower_band"]    = pd.Series(_lag1_float(lower_band),    index=idx)
    out["rolling_midpoint_bounce_range"]         = pd.Series(_lag1_float(rolling_range), index=idx)
    out["rolling_midpoint_bounce_dev_norm"]      = pd.Series(_lag1_float(deviation_norm), index=idx)
    out["rolling_midpoint_bounce_smoothed_dev"]  = pd.Series(_lag1_float(smoothed_dev),  index=idx)

    return out