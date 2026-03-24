"""
features/donchian_breakout_momentum.py - Donchian Channel Breakout with Volume Surge confirmation.
Computes Donchian channel breakout signals confirmed by volume above a rolling average multiple.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_donchian_breakout_momentum_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Donchian Channel Breakout with Volume Surge features.

    Logic:
      - Donchian Upper = highest high over 'period' bars
      - Donchian Lower = lowest low over 'period' bars
      - Donchian Mid   = (upper + lower) / 2
      - Volume surge   = current volume > vol_mult * rolling_mean(volume, vol_period)

      bull: Close breaks above Donchian Upper AND volume surge (bullish breakout)
      bear: Close breaks below Donchian Lower AND volume surge (bearish breakout)

    Additional outputs:
      donchian_breakout_momentum_upper     : Donchian upper band (shift 1)
      donchian_breakout_momentum_lower     : Donchian lower band (shift 1)
      donchian_breakout_momentum_mid       : Donchian midline (shift 1)
      donchian_breakout_momentum_vol_surge : volume surge flag (shift 1)
      donchian_breakout_momentum_bull      : bullish breakout signal (shift 1)
      donchian_breakout_momentum_bear      : bearish breakout signal (shift 1)

    Args:
        df:     OHLCV DataFrame with columns Open, High, Low, Close, Volume
        config: Full config dict. Reads config['tv_indicators']['donchian_breakout_momentum'].

    Returns:
        df copy with donchian_breakout_momentum_ columns added.
    """
    params = config.get('tv_indicators', {}).get('donchian_breakout_momentum', {})
    period     = int(params.get('period',     8))
    vol_period = int(params.get('vol_period', 6))
    vol_mult   = float(params.get('vol_mult', 1.5))

    # Clamp periods to scalping-safe range
    period     = max(3, min(period,     14))
    vol_period = max(3, min(vol_period, 12))

    out    = df.copy()
    high   = out['High'].astype(float).values
    low    = out['Low'].astype(float).values
    close  = out['Close'].astype(float).values
    volume = out['Volume'].astype(float).values

    # Donchian Upper: highest high over 'period' bars (talib MAX)
    upper = talib.MAX(high, timeperiod=period)

    # Donchian Lower: lowest low over 'period' bars (talib MIN)
    lower = talib.MIN(low, timeperiod=period)

    # Donchian Midline
    mid = (upper + lower) / 2.0

    # Rolling volume mean over vol_period bars using talib MA
    vol_ma = talib.SMA(volume, timeperiod=vol_period)

    # Volume surge flag: current volume > vol_mult * rolling mean
    vol_surge = volume > (vol_mult * vol_ma)

    # Breakout signals (raw, before shift)
    # Bull: close crosses above upper band with volume confirmation
    bull_raw = (close > upper) & vol_surge

    # Bear: close crosses below lower band with volume confirmation
    bear_raw = (close < lower) & vol_surge

    # Helper: numpy-level 1-bar lag
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    def _lag1_float(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = np.nan
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    out['donchian_breakout_momentum_upper']     = pd.Series(_lag1_float(upper),     index=idx, dtype=float)
    out['donchian_breakout_momentum_lower']     = pd.Series(_lag1_float(lower),     index=idx, dtype=float)
    out['donchian_breakout_momentum_mid']       = pd.Series(_lag1_float(mid),       index=idx, dtype=float)
    out['donchian_breakout_momentum_vol_surge'] = pd.Series(_lag1(vol_surge),       index=idx, dtype=bool)
    out['donchian_breakout_momentum_bull']      = pd.Series(_lag1(bull_raw),        index=idx, dtype=bool)
    out['donchian_breakout_momentum_bear']      = pd.Series(_lag1(bear_raw),        index=idx, dtype=bool)

    return out