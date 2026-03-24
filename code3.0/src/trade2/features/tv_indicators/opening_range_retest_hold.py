"""
features/opening_range_retest_hold.py - Opening Range Retest & Hold feature detection.
Detects when price breaks out of the opening range, retests the boundary, and resumes.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_opening_range_retest_hold_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Opening Range Retest & Hold features for scalping on 5M charts.

    Logic:
      1. Define opening range as the High/Low over the first `range_bars` bars of a rolling window.
      2. Detect breakouts above range_high or below range_low.
      3. After a breakout, detect a retest of the broken level within `retest_bars`.
      4. Bull signal: price broke above range_high, retested it (came back near/below), then resumed up.
      5. Bear signal: price broke below range_low, retested it (came back near/above), then resumed down.

    Params (from config):
      range_bars:   bars to define the opening range (default 5)
      retest_bars:  bars to look back for the retest after breakout (default 2)

    Args:
        df:     OHLCV DataFrame with columns Open, High, Low, Close, Volume
        config: Full config dict.

    Returns:
        df copy with opening_range_retest_hold_ columns added.
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df

    params = config.get('tv_indicators', {}).get('opening_range_retest_hold', {})
    range_bars  = int(params.get('range_bars',  5))
    retest_bars = int(params.get('retest_bars', 2))

    # Clamp to scalping-safe ranges
    range_bars  = max(3, min(range_bars,  12))
    retest_bars = max(1, min(retest_bars,  5))

    result = df.copy()
    
    try:
        close = result['Close'].astype(float).values
        high  = result['High'].astype(float).values
        low   = result['Low'].astype(float).values
        n     = len(close)

        # ── Rolling opening-range High / Low ────────────────────────────────────
        range_high = talib.MAX(high, timeperiod=range_bars)
        range_low  = talib.MIN(low,  timeperiod=range_bars)

        # ── ATR for proximity threshold (touch tolerance) ────────────────────────
        atr_period = max(3, range_bars)
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        tolerance_frac = 0.25
        tol = atr * tolerance_frac

        # ── Breakout detection ───────────────────────────────────────────────────
        bull_breakout = np.zeros(n, dtype=bool)
        bear_breakout = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(range_high[i - 1]) or np.isnan(range_low[i - 1]):
                continue
            bull_breakout[i] = (close[i] > range_high[i - 1]) and (close[i - 1] <= range_high[i - 1])
            bear_breakout[i] = (close[i] < range_low[i - 1])  and (close[i - 1] >= range_low[i - 1])

        # ── Retest & Hold detection ──────────────────────────────────────────────
        bull_signal = np.zeros(n, dtype=bool)
        bear_signal = np.zeros(n, dtype=bool)

        last_bull_break = -999
        last_bear_break = -999
        bull_break_level = np.nan
        bear_break_level = np.nan

        for i in range(1, n):
            if bull_breakout[i]:
                last_bull_break  = i
                bull_break_level = range_high[i - 1]
            if bear_breakout[i]:
                last_bear_break  = i
                bear_break_level = range_low[i - 1]

            tol_i = tol[i] if not np.isnan(tol[i]) else 0.0

            if (0 < i - last_bull_break <= retest_bars) and not np.isnan(bull_break_level):
                touched = low[i] <= bull_break_level + tol_i
                held    = close[i] > bull_break_level - tol_i
                if touched and held:
                    bull_signal[i] = True

            if (0 < i - last_bear_break <= retest_bars) and not np.isnan(bear_break_level):
                touched = high[i] >= bear_break_level - tol_i
                held    = close[i] < bear_break_level + tol_i
                if touched and held:
                    bear_signal[i] = True

        # ── Momentum confirmation via short EMA slope ────────────────────────────
        ema_fast = talib.EMA(close, timeperiod=3)
        ema_slow = talib.EMA(close, timeperiod=8)

        ema_bull = ema_fast > ema_slow
        ema_bear = ema_fast < ema_slow

        bull_final = bull_signal & ema_bull
        bear_final = bear_signal & ema_bear

        # ── Lag-1 helper ────────────────────────────────────────────────────────
        def _lag1(arr: np.ndarray) -> np.ndarray:
            out_arr = np.empty(len(arr), dtype=bool)
            out_arr[0] = False
            out_arr[1:] = arr[:-1]
            return out_arr

        idx = result.index

        result['opening_range_retest_hold_bull']          = pd.Series(_lag1(bull_final),    index=idx, dtype=bool)
        result['opening_range_retest_hold_bear']          = pd.Series(_lag1(bear_final),    index=idx, dtype=bool)
        result['opening_range_retest_hold_bull_breakout'] = pd.Series(_lag1(bull_breakout), index=idx, dtype=bool)
        result['opening_range_retest_hold_bear_breakout'] = pd.Series(_lag1(bear_breakout), index=idx, dtype=bool)

    except Exception:
        result['opening_range_retest_hold_bull']          = False
        result['opening_range_retest_hold_bear']          = False
        result['opening_range_retest_hold_bull_breakout'] = False
        result['opening_range_retest_hold_bear_breakout'] = False

    return result