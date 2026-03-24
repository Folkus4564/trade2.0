"""
features/opening_range_midpoint_reject.py - Opening Range Midpoint Rejection feature detection.
Detects price failures beyond one side of the opening range that rotate around the midpoint.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_opening_range_midpoint_reject_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Opening Range Midpoint Rejection features.

    Logic:
      - Opening range is defined as the High/Low over the first `range_bars` bars of each session
        (or a rolling window of `range_bars` bars for non-session data).
      - Midpoint = (range_high + range_low) / 2
      - Bull signal: price pushed below range_low (false breakdown), then closes back above midpoint
        within `confirm_bars` bars -> bullish rejection
      - Bear signal: price pushed above range_high (false breakout), then closes back below midpoint
        within `confirm_bars` bars -> bearish rejection

    Columns added (all shift(1)):
      opening_range_midpoint_reject_range_high  : rolling high over range_bars
      opening_range_midpoint_reject_range_low   : rolling low over range_bars
      opening_range_midpoint_reject_midpoint    : midpoint of range
      opening_range_midpoint_reject_bull        : bullish midpoint rejection signal
      opening_range_midpoint_reject_bear        : bearish midpoint rejection signal

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config['tv_indicators']['opening_range_midpoint_reject'].

    Returns:
        df copy with opening_range_midpoint_reject_ columns added.
    """
    params = config.get('tv_indicators', {}).get('opening_range_midpoint_reject', {})
    confirm_bars = int(params.get('confirm_bars', 1))
    range_bars   = int(params.get('range_bars', 5))

    # Clamp to scalping-safe ranges
    range_bars   = max(3, min(range_bars,   12))
    confirm_bars = max(1, min(confirm_bars,  5))

    out   = df.copy()
    close = out['Close'].astype(float).values
    high  = out['High'].astype(float).values
    low   = out['Low'].astype(float).values
    n     = len(close)

    # ------------------------------------------------------------------
    # Rolling opening range: highest high and lowest low over range_bars
    # Using talib MAX/MIN for efficiency
    # ------------------------------------------------------------------
    range_high = talib.MAX(high, timeperiod=range_bars)
    range_low  = talib.MIN(low,  timeperiod=range_bars)
    midpoint   = (range_high + range_low) / 2.0

    # ------------------------------------------------------------------
    # Rejection logic (no lookahead: all computed on bar-close values)
    #
    # Bull rejection (false breakdown + recover above midpoint):
    #   Step 1: within the last (range_bars) bars, the LOW touched below range_low
    #           (i.e. min(low[i - range_bars : i]) < range_low[i])
    #   Step 2: current close is ABOVE midpoint
    #   Step 3: previous close was at or below midpoint (fresh cross)
    #
    # Bear rejection (false breakout + recover below midpoint):
    #   Step 1: within the last (range_bars) bars, the HIGH touched above range_high
    #           (i.e. max(high[i - range_bars : i]) > range_high[i])
    #   Step 2: current close is BELOW midpoint
    #   Step 3: previous close was at or above midpoint (fresh cross)
    # ------------------------------------------------------------------

    # Rolling extreme lows/highs over confirm_bars to detect the touch
    confirm_period = max(2, confirm_bars + 1)  # look back slightly further for the touch
    touch_low_window  = talib.MIN(low,  timeperiod=confirm_period)   # lowest low recently
    touch_high_window = talib.MAX(high, timeperiod=confirm_period)   # highest high recently

    # False breakdown: low pierced below range_low in recent bars
    false_breakdown = touch_low_window < range_low
    # False breakout:  high pierced above range_high in recent bars
    false_breakout  = touch_high_window > range_high

    # Close relative to midpoint
    above_mid = close > midpoint
    below_mid = close < midpoint

    # Previous-bar midpoint cross (numpy shift)
    def _lag1_bool(arr: np.ndarray) -> np.ndarray:
        result = np.empty(len(arr), dtype=bool)
        result[0] = False
        result[1:] = arr[:-1]
        return result

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        result = np.empty(len(arr), dtype=float)
        result[0] = np.nan
        result[1:] = arr[:-1]
        return result

    prev_above_mid = _lag1_bool(above_mid)
    prev_below_mid = _lag1_bool(below_mid)

    # Bull: false breakdown occurred AND price just crossed back above midpoint
    bull_raw = false_breakdown & above_mid & ~prev_above_mid

    # Bear: false breakout occurred AND price just crossed back below midpoint
    bear_raw = false_breakout & below_mid & ~prev_below_mid

    # Handle NaN regions from talib (first range_bars-1 bars are NaN)
    valid = ~np.isnan(range_high) & ~np.isnan(range_low) & ~np.isnan(touch_low_window) & ~np.isnan(touch_high_window)
    bull_raw = bull_raw & valid
    bear_raw = bear_raw & valid

    # ------------------------------------------------------------------
    # Shift(1) all outputs for lag safety
    # ------------------------------------------------------------------
    idx = out.index

    out['opening_range_midpoint_reject_range_high'] = pd.Series(_lag1_float(range_high), index=idx)
    out['opening_range_midpoint_reject_range_low']  = pd.Series(_lag1_float(range_low),  index=idx)
    out['opening_range_midpoint_reject_midpoint']   = pd.Series(_lag1_float(midpoint),   index=idx)
    out['opening_range_midpoint_reject_bull']       = pd.Series(_lag1_bool(bull_raw),    index=idx, dtype=bool)
    out['opening_range_midpoint_reject_bear']       = pd.Series(_lag1_bool(bear_raw),    index=idx, dtype=bool)

    return out