"""
features/pivot_high_low_burst.py - Pivot High/Low Burst feature detection.
Detects breakouts through recent pivot highs/lows with momentum acceleration.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_pivot_high_low_burst_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Pivot High/Low Burst features for scalping on 5M charts.

    Logic:
      - Identify pivot highs and pivot lows using left/right bar lookback.
      - A pivot high is a bar whose high is the highest in [i-pivot_left .. i+pivot_right].
      - A pivot low  is a bar whose low  is the lowest  in [i-pivot_left .. i+pivot_right].
      - Track the most recent confirmed pivot high and pivot low levels.
      - A bull burst occurs when Close breaks above the most recent pivot high level
        AND momentum (close - close[confirm_bars] bars ago) is positive (acceleration).
      - A bear burst occurs when Close breaks below the most recent pivot low level
        AND momentum is negative.
      - confirm_bars controls how many bars back we check for acceleration.

    Outputs (all shift(1)):
      pivot_high_low_burst_pivot_high   : most recent pivot high level
      pivot_high_low_burst_pivot_low    : most recent pivot low level
      pivot_high_low_burst_momentum     : short-term momentum (close - close[confirm_bars])
      pivot_high_low_burst_bull         : True when bullish burst (breakout above pivot high with +momentum)
      pivot_high_low_burst_bear         : True when bearish burst (breakout below pivot low with -momentum)

    Args:
        df:     OHLCV DataFrame with columns Open, High, Low, Close, Volume
        config: Full config dict. Reads config['tv_indicators']['pivot_high_low_burst'].

    Returns:
        df copy with pivot_high_low_burst_ columns added.
    """
    cfg = config.get('tv_indicators', {}).get('pivot_high_low_burst', {})
    pivot_left   = int(cfg.get('pivot_left',   2))
    pivot_right  = int(cfg.get('pivot_right',  2))
    confirm_bars = int(cfg.get('confirm_bars', 1))

    # Clamp to scalping-safe ranges
    pivot_left   = max(1, min(pivot_left,   6))
    pivot_right  = max(1, min(pivot_right,  6))
    confirm_bars = max(1, min(confirm_bars, 5))

    out   = df.copy()
    n     = len(out)
    high  = out['High'].astype(float).values
    low   = out['Low'].astype(float).values
    close = out['Close'].astype(float).values

    # ------------------------------------------------------------------
    # Step 1: Detect pivot highs and pivot lows (no lookahead).
    #
    # A pivot high at bar i is confirmed only after pivot_right bars have
    # passed, i.e. at bar i + pivot_right. We mark pivot highs/lows at
    # the confirmation bar index to avoid lookahead.
    # ------------------------------------------------------------------
    pivot_high_vals = np.full(n, np.nan)   # confirmed pivot high prices
    pivot_low_vals  = np.full(n, np.nan)   # confirmed pivot low prices

    window = pivot_left + pivot_right + 1   # total bars in pivot window

    for i in range(window - 1, n):
        # The candidate pivot bar is pivot_right bars before current bar i
        candidate = i - pivot_right

        # High pivot: candidate high is the highest in the window
        window_high = high[candidate - pivot_left: i + 1]   # length = window
        if high[candidate] == np.max(window_high):
            # Confirmed at bar i (pivot_right bars after candidate)
            pivot_high_vals[i] = high[candidate]

        # Low pivot: candidate low is the lowest in the window
        window_low = low[candidate - pivot_left: i + 1]
        if low[candidate] == np.min(window_low):
            pivot_low_vals[i] = low[candidate]

    # ------------------------------------------------------------------
    # Step 2: Forward-fill to get the most recent confirmed pivot level
    #         at each bar (carry forward until a new pivot appears).
    # ------------------------------------------------------------------
    last_pivot_high = np.full(n, np.nan)
    last_pivot_low  = np.full(n, np.nan)

    running_ph = np.nan
    running_pl = np.nan
    for i in range(n):
        if not np.isnan(pivot_high_vals[i]):
            running_ph = pivot_high_vals[i]
        if not np.isnan(pivot_low_vals[i]):
            running_pl = pivot_low_vals[i]
        last_pivot_high[i] = running_ph
        last_pivot_low[i]  = running_pl

    # ------------------------------------------------------------------
    # Step 3: Short-term momentum = close - close[confirm_bars ago]
    #         Using talib ROCP isn't ideal; simple difference is cleaner.
    # ------------------------------------------------------------------
    momentum = np.full(n, np.nan)
    for i in range(confirm_bars, n):
        momentum[i] = close[i] - close[i - confirm_bars]

    # ------------------------------------------------------------------
    # Step 4: Bull / Bear burst signals (raw, before shift)
    #
    # Bull burst: close > last confirmed pivot high AND momentum > 0
    # Bear burst: close < last confirmed pivot low  AND momentum < 0
    # ------------------------------------------------------------------
    bull_raw = np.zeros(n, dtype=bool)
    bear_raw = np.zeros(n, dtype=bool)

    for i in range(n):
        if (not np.isnan(last_pivot_high[i])
                and not np.isnan(momentum[i])
                and close[i] > last_pivot_high[i]
                and momentum[i] > 0.0):
            bull_raw[i] = True
        if (not np.isnan(last_pivot_low[i])
                and not np.isnan(momentum[i])
                and close[i] < last_pivot_low[i]
                and momentum[i] < 0.0):
            bear_raw[i] = True

    # ------------------------------------------------------------------
    # Step 5: Shift(1) all outputs for lag safety
    # ------------------------------------------------------------------
    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty(len(arr), dtype=float)
        out_arr[0] = np.nan
        out_arr[1:] = arr[:-1]
        return out_arr

    def _lag1_bool(arr: np.ndarray) -> np.ndarray:
        out_arr = np.zeros(len(arr), dtype=bool)
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index
    out['pivot_high_low_burst_pivot_high'] = pd.Series(_lag1_float(last_pivot_high), index=idx, dtype=float)
    out['pivot_high_low_burst_pivot_low']  = pd.Series(_lag1_float(last_pivot_low),  index=idx, dtype=float)
    out['pivot_high_low_burst_momentum']   = pd.Series(_lag1_float(momentum),        index=idx, dtype=float)
    out['pivot_high_low_burst_bull']       = pd.Series(_lag1_bool(bull_raw),         index=idx, dtype=bool)
    out['pivot_high_low_burst_bear']       = pd.Series(_lag1_bool(bear_raw),         index=idx, dtype=bool)

    return out