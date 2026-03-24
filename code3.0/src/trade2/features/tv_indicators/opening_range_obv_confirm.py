"""
features/opening_range_obv_confirm.py - Opening Range OBV Confirm feature detection.
OBV makes a confirming new intraday swing as price exits opening range.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_opening_range_obv_confirm_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Opening Range OBV Confirm features.

    Logic:
      1. Define the opening range as the High/Low over the first `range_bars` bars
         of each session (rolling window of `range_bars` bars used as proxy).
      2. Compute OBV.
      3. Track OBV swing high/low over `obv_lookback` bars.
      4. Bull signal: price breaks above opening range high AND OBV makes a new
         swing high (OBV > max OBV over lookback) confirming the breakout.
      5. Bear signal: price breaks below opening range low AND OBV makes a new
         swing low (OBV < min OBV over lookback) confirming the breakdown.

    All outputs are shift(1) for lag safety.

    Args:
        df:     OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        config: Full config dict.

    Returns:
        df copy with opening_range_obv_confirm_ columns added.
    """
    params = config.get('tv_indicators', {}).get('opening_range_obv_confirm', {})
    obv_lookback = int(params.get('obv_lookback', 10))
    range_bars   = int(params.get('range_bars', 5))

    # Clamp to scalping-safe ranges
    obv_lookback = max(3, min(obv_lookback, 14))
    range_bars   = max(3, min(range_bars, 12))

    out    = df.copy()
    close  = out['Close'].astype(float).values
    high   = out['High'].astype(float).values
    low    = out['Low'].astype(float).values
    volume = out['Volume'].astype(float).values

    n = len(close)

    # ── 1. OBV via talib ──────────────────────────────────────────────────────
    obv = talib.OBV(close, volume)  # shape (n,)

    # ── 2. Opening range: rolling High/Low over `range_bars` bars ────────────
    #    We treat a rolling window as the "session opening range" for scalping.
    range_high = pd.Series(high).rolling(range_bars, min_periods=range_bars).max().values
    range_low  = pd.Series(low).rolling(range_bars, min_periods=range_bars).min().values

    # ── 3. OBV swing confirmation ─────────────────────────────────────────────
    #    New OBV swing high: current OBV > max OBV over previous `obv_lookback` bars
    #    New OBV swing low:  current OBV < min OBV over previous `obv_lookback` bars
    #    Use shift(1) on OBV window so we don't include current bar in lookback max/min
    obv_series      = pd.Series(obv)
    obv_roll_max    = obv_series.shift(1).rolling(obv_lookback, min_periods=obv_lookback).max().values
    obv_roll_min    = obv_series.shift(1).rolling(obv_lookback, min_periods=obv_lookback).min().values

    obv_new_high = obv > obv_roll_max   # OBV breaks to new swing high
    obv_new_low  = obv < obv_roll_min   # OBV breaks to new swing low

    # ── 4. Price breakout from opening range ──────────────────────────────────
    price_above_range = close > range_high   # bullish breakout
    price_below_range = close < range_low    # bearish breakdown

    # ── 5. Combined signals (raw, before final shift) ─────────────────────────
    bull_raw = price_above_range & obv_new_high
    bear_raw = price_below_range & obv_new_low

    # ── 6. OBV smoothed (short EMA for trend context) ─────────────────────────
    obv_ema_fast = talib.EMA(obv, timeperiod=3)
    obv_ema_slow = talib.EMA(obv, timeperiod=8)

    # OBV trend: fast EMA above slow EMA → bullish OBV momentum
    obv_trend_bull = obv_ema_fast > obv_ema_slow
    obv_trend_bear = obv_ema_fast < obv_ema_slow

    # Refine signals with OBV trend filter
    bull_raw = bull_raw & obv_trend_bull
    bear_raw = bear_raw & obv_trend_bear

    # ── 7. Shift(1) for lag safety ────────────────────────────────────────────
    def _lag1_bool(arr: np.ndarray) -> np.ndarray:
        out_arr = np.zeros(len(arr), dtype=bool)
        out_arr[1:] = arr[:-1]
        return out_arr

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        out_arr = np.full(len(arr), np.nan)
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    out['opening_range_obv_confirm_bull']         = pd.Series(_lag1_bool(bull_raw),        index=idx, dtype=bool)
    out['opening_range_obv_confirm_bear']         = pd.Series(_lag1_bool(bear_raw),        index=idx, dtype=bool)
    out['opening_range_obv_confirm_obv']          = pd.Series(_lag1_float(obv),            index=idx, dtype=float)
    out['opening_range_obv_confirm_obv_ema_fast'] = pd.Series(_lag1_float(obv_ema_fast),   index=idx, dtype=float)
    out['opening_range_obv_confirm_obv_ema_slow'] = pd.Series(_lag1_float(obv_ema_slow),   index=idx, dtype=float)
    out['opening_range_obv_confirm_range_high']   = pd.Series(_lag1_float(range_high),     index=idx, dtype=float)
    out['opening_range_obv_confirm_range_low']    = pd.Series(_lag1_float(range_low),      index=idx, dtype=float)
    out['opening_range_obv_confirm_obv_new_high'] = pd.Series(_lag1_bool(obv_new_high),    index=idx, dtype=bool)
    out['opening_range_obv_confirm_obv_new_low']  = pd.Series(_lag1_bool(obv_new_low),     index=idx, dtype=bool)

    return out