"""
features/median_price_thrust_break.py - Median Price Thrust Break feature detection.
Computes breakout signals confirmed by median price acceleration and close beyond recent structure.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_median_price_thrust_break_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Median Price Thrust Break features.

    Concept:
      - Median price = (High + Low) / 2
      - Thrust = rate of change / acceleration of median price over short period
      - Structure break = close beyond recent high/low swing points
      - Bull signal: median price thrusting upward AND close breaks above recent high structure
      - Bear signal: median price thrusting downward AND close breaks below recent low structure

    Columns added:
      median_price_thrust_break_median     : (H+L)/2
      median_price_thrust_break_thrust     : ROC of median price (thrust magnitude)
      median_price_thrust_break_accel      : acceleration (ROC of ROC)
      median_price_thrust_break_struct_high: recent structure high (rolling max of highs)
      median_price_thrust_break_struct_low : recent structure low  (rolling min of lows)
      median_price_thrust_break_bull       : bullish breakout signal (shift(1))
      median_price_thrust_break_bear       : bearish breakout signal (shift(1))

    Args:
        df:     OHLCV DataFrame with Open, High, Low, Close, Volume
        config: Full config dict. Reads config['tv_indicators']['median_price_thrust_break'].

    Returns:
        df copy with median_price_thrust_break_ columns added.
    """
    params = config.get('tv_indicators', {}).get('median_price_thrust_break', {})
    period           = int(params.get('period', 6))
    thrust_threshold = float(params.get('thrust_threshold', 0.25))

    # Clamp to scalping-safe range
    period = max(3, min(period, 14))

    # Structure lookback: slightly wider than thrust period, still short
    struct_period = min(period * 2, 12)

    out   = df.copy()
    close = out['Close'].astype(float).values
    high  = out['High'].astype(float).values
    low   = out['Low'].astype(float).values

    n = len(close)

    # --- Median Price ---
    median = (high + low) / 2.0

    # --- Thrust: ROC of median price over `period` bars ---
    # ROC = ((median[i] - median[i-period]) / median[i-period]) * 100
    thrust = talib.ROC(median, timeperiod=period)

    # --- Acceleration: ROC of thrust (momentum of momentum) ---
    # Use a short EMA to smooth thrust first, then ROC of that
    thrust_smooth = talib.EMA(thrust, timeperiod=max(3, period // 2))
    accel = talib.ROC(thrust_smooth, timeperiod=max(3, period // 2))

    # --- Structure High/Low: rolling max/min of highs/lows over struct_period ---
    # Use pandas rolling for this (no direct talib equivalent for rolling high/low)
    high_s  = pd.Series(high)
    low_s   = pd.Series(low)
    close_s = pd.Series(close)

    # Shift by 1 before rolling to avoid current bar's own value in structure
    struct_high = high_s.shift(1).rolling(window=struct_period, min_periods=period).max().values
    struct_low  = low_s.shift(1).rolling(window=struct_period, min_periods=period).min().values

    # --- Bull signal (raw) ---
    # Conditions:
    #   1. Median price thrust > threshold (upward acceleration)
    #   2. Acceleration is positive (thrust is increasing)
    #   3. Close breaks above recent structure high
    bull_raw = (
        (thrust > thrust_threshold) &
        (accel > 0.0) &
        (close > struct_high)
    )

    # --- Bear signal (raw) ---
    # Conditions:
    #   1. Median price thrust < -threshold (downward acceleration)
    #   2. Acceleration is negative (thrust is decreasing / more negative)
    #   3. Close breaks below recent structure low
    bear_raw = (
        (thrust < -thrust_threshold) &
        (accel < 0.0) &
        (close < struct_low)
    )

    # Replace NaN-driven False with explicit False
    def _safe_bool(arr: np.ndarray) -> np.ndarray:
        result = np.zeros(len(arr), dtype=bool)
        valid  = ~np.isnan(arr.astype(float)) if arr.dtype != bool else np.ones(len(arr), dtype=bool)
        result[valid] = arr[valid]
        return result

    bull_raw = _safe_bool(bull_raw.astype(float))
    bear_raw = _safe_bool(bear_raw.astype(float))

    # --- Lag helper ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.zeros(len(arr), dtype=bool)
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    # --- Store outputs (all shift(1)) ---
    out['median_price_thrust_break_median']      = pd.Series(median,       index=idx).shift(1)
    out['median_price_thrust_break_thrust']      = pd.Series(thrust,       index=idx).shift(1)
    out['median_price_thrust_break_accel']       = pd.Series(accel,        index=idx).shift(1)
    out['median_price_thrust_break_struct_high'] = pd.Series(struct_high,  index=idx).shift(1)
    out['median_price_thrust_break_struct_low']  = pd.Series(struct_low,   index=idx).shift(1)
    out['median_price_thrust_break_bull']        = pd.Series(_lag1(bull_raw), index=idx, dtype=bool)
    out['median_price_thrust_break_bear']        = pd.Series(_lag1(bear_raw), index=idx, dtype=bool)

    return out