"""
features/opening_range_high_low_sweep.py - Opening Range High/Low Sweep feature detection.
Detects when price sweeps an opening range boundary and expands sharply beyond it.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_opening_range_high_low_sweep_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Opening Range High/Low Sweep features.

    Logic:
      - Define a rolling opening range over 'range_bars' bars (high/low of that window)
      - A HIGH SWEEP (bull) occurs when:
          * Price previously touched/pierced the range high (within sweep_tolerance)
          * Then closes ABOVE the range high (breakout expansion)
          * Momentum confirms (close > open, body expansion)
      - A LOW SWEEP (bear) occurs when:
          * Price previously touched/pierced the range low (within sweep_tolerance)
          * Then closes BELOW the range low (breakout expansion)
          * Momentum confirms (close < open, body expansion)

    Signals (all shift(1) for lag safety):
      opening_range_high_low_sweep_bull: bullish sweep (price swept range high and expanded)
      opening_range_high_low_sweep_bear: bearish sweep (price swept range low and expanded)

    Args:
        df:     OHLCV DataFrame with columns Open, High, Low, Close, Volume
        config: Full config dict. Reads config['tv_indicators']['opening_range_high_low_sweep'].

    Returns:
        df copy with opening_range_high_low_sweep_ columns added.
    """
    params = config.get('tv_indicators', {}).get('opening_range_high_low_sweep', {})
    range_bars       = int(params.get('range_bars', 5))
    sweep_tolerance  = float(params.get('sweep_tolerance', 0.1))

    # Clamp to scalping-friendly range
    range_bars  = max(3, min(range_bars, 12))

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values
    open_ = out["Open"].astype(float).values
    n     = len(close)

    # --- Rolling opening range high/low (over range_bars bars) ---
    # Use a shifted window so range is defined from PAST bars only (no lookahead)
    # range_high[i] = max(high[i-range_bars : i])
    # range_low[i]  = min(low[i-range_bars  : i])
    close_s = pd.Series(close)
    high_s  = pd.Series(high)
    low_s   = pd.Series(low)

    range_high = high_s.shift(1).rolling(window=range_bars, min_periods=range_bars).max().values
    range_low  = low_s.shift(1).rolling(window=range_bars, min_periods=range_bars).min().values

    # --- ATR for dynamic tolerance (short period for scalping) ---
    atr_period = max(3, min(range_bars, 7))
    atr = talib.ATR(high, low, close, timeperiod=atr_period)

    # --- Bar body size ---
    body = np.abs(close - open_)

    # --- Sweep detection logic ---
    # Tolerance band: sweep_tolerance * ATR
    # A sweep touch means the bar's low dipped into/below range_high - tol (for high sweep)
    # or bar's high rose into/above range_low + tol (for low sweep)

    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)

    for i in range(range_bars + 1, n):
        rh = range_high[i]
        rl = range_low[i]
        at = atr[i]

        if np.isnan(rh) or np.isnan(rl) or np.isnan(at) or at <= 0:
            continue

        tol = sweep_tolerance * at

        # --- BULL SWEEP: low swept below/to range_high, then closes above range_high ---
        # Bar dipped near or into range high zone (fakeout / liquidity grab below)
        # then closed above range_high (expansion)
        low_swept_range_high  = low[i] <= (rh + tol)       # bar touched/pierced range high area
        close_above_range_high = close[i] > rh              # close is above range high
        bull_body_confirm      = close[i] > open_[i]        # bullish bar
        body_expansion_bull    = body[i] > 0.3 * at         # meaningful body size

        bull_sweep[i] = (
            low_swept_range_high
            and close_above_range_high
            and bull_body_confirm
            and body_expansion_bull
        )

        # --- BEAR SWEEP: high swept above/to range_low, then closes below range_low ---
        # Bar spiked into range low zone (fakeout / liquidity grab above)
        # then closed below range_low (expansion)
        high_swept_range_low  = high[i] >= (rl - tol)      # bar touched/pierced range low area
        close_below_range_low  = close[i] < rl             # close is below range low
        bear_body_confirm      = close[i] < open_[i]       # bearish bar
        body_expansion_bear    = body[i] > 0.3 * at        # meaningful body size

        bear_sweep[i] = (
            high_swept_range_low
            and close_below_range_low
            and bear_body_confirm
            and body_expansion_bear
        )

    # --- Momentum confirmation: use short EMA slope ---
    ema_fast = talib.EMA(close, timeperiod=3)
    ema_slow = talib.EMA(close, timeperiod=7)

    # Momentum filter: fast EMA above slow = bullish momentum, below = bearish
    mom_bull = ema_fast > ema_slow
    mom_bear = ema_fast < ema_slow

    # Combine sweep with momentum
    bull_signal = bull_sweep & mom_bull
    bear_signal = bear_sweep & mom_bear

    # --- Lag helper: 1-bar shift ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty(len(arr), dtype=bool)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    # All outputs shift(1) for lag safety
    out["opening_range_high_low_sweep_bull"]        = pd.Series(_lag1(bull_signal),   index=idx, dtype=bool)
    out["opening_range_high_low_sweep_bear"]        = pd.Series(_lag1(bear_signal),   index=idx, dtype=bool)
    out["opening_range_high_low_sweep_range_high"]  = pd.Series(range_high,           index=idx).shift(1)
    out["opening_range_high_low_sweep_range_low"]   = pd.Series(range_low,            index=idx).shift(1)
    out["opening_range_high_low_sweep_atr"]         = pd.Series(atr,                  index=idx).shift(1)

    return out