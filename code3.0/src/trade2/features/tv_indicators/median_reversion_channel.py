"""
features/median_reversion_channel.py - Median Reversion Channel feature detection.
Computes a channel based on median price and ATR bands, detecting mean reversion signals.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_median_reversion_channel_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Median Reversion Channel features for scalping on 5M charts.

    The channel is built around a median-based midline using ATR bands.
    - Midline: EMA of typical price (HL2 median approximation)
    - Upper/Lower bands: midline +/- band_mult * ATR

    Signals:
      bull: price touched or pierced lower band and is now closing back above it
            (oversold bounce / mean reversion long)
      bear: price touched or pierced upper band and is now closing back below it
            (overbought reversal / mean reversion short)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["median_reversion_channel"].

    Returns:
        df copy with median_reversion_channel_ columns added.
    """
    cfg = config.get('tv_indicators', {}).get('median_reversion_channel', {})
    period    = int(cfg.get('period', 12))
    band_mult = float(cfg.get('band_mult', 1.3))

    # Clamp period to scalping range
    period = max(3, min(period, 14))

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values
    open_ = out["Open"].astype(float).values

    # Median source: HL2
    hl2 = (high + low) / 2.0

    # Midline: EMA of HL2 (acts as median channel center)
    midline = talib.EMA(hl2, timeperiod=period)

    # ATR for dynamic band width
    atr = talib.ATR(high, low, close, timeperiod=period)

    # Channel bands
    upper_band = midline + band_mult * atr
    lower_band = midline - band_mult * atr

    # Channel width and percent position of close within channel
    channel_width = upper_band - lower_band
    # Avoid division by zero
    channel_width_safe = np.where(channel_width > 1e-10, channel_width, np.nan)
    pct_pos = (close - lower_band) / channel_width_safe  # 0=at lower, 1=at upper

    # Touch detection: price reached or pierced band on previous bar
    # Lower band touch: low <= lower_band
    # Upper band touch: high >= upper_band
    lower_touch = low <= lower_band
    upper_touch = high >= upper_band

    # Reversion confirmation: close returned inside channel
    close_above_lower = close > lower_band
    close_below_upper = close < upper_band

    # Bull signal: previous bar touched lower band AND current close is back above lower band
    # Use a short lookback (3 bars) for touch memory to catch scalp entries
    lookback = 3

    n = len(close)
    lower_touch_recent = np.zeros(n, dtype=bool)
    upper_touch_recent = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        lower_touch_recent[i] = np.any(lower_touch[i - lookback:i])
        upper_touch_recent[i] = np.any(upper_touch[i - lookback:i])

    # Raw signals (before shift)
    # Bull: recent lower band touch + close back above lower band + close below midline (hasn't overshot)
    bull_raw = lower_touch_recent & close_above_lower & (close <= midline)

    # Bear: recent upper band touch + close back below upper band + close above midline
    bear_raw = upper_touch_recent & close_below_upper & (close >= midline)

    # Additional momentum confirmation using short EMA crossover
    fast_ema = talib.EMA(close, timeperiod=max(3, period // 3))
    slow_ema = talib.EMA(close, timeperiod=period)

    bull_momentum = fast_ema > slow_ema
    bear_momentum = fast_ema < slow_ema

    # Combine with momentum for higher quality signals
    bull_final = bull_raw & bull_momentum
    bear_final = bear_raw & bear_momentum

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

    # Boolean signals (shift(1) for lag safety)
    out["median_reversion_channel_bull"] = pd.Series(_lag1(bull_final), index=idx, dtype=bool)
    out["median_reversion_channel_bear"] = pd.Series(_lag1(bear_final), index=idx, dtype=bool)

    # Supplementary float columns (shift(1))
    out["median_reversion_channel_midline"]    = pd.Series(_lag1_float(midline),    index=idx)
    out["median_reversion_channel_upper"]      = pd.Series(_lag1_float(upper_band), index=idx)
    out["median_reversion_channel_lower"]      = pd.Series(_lag1_float(lower_band), index=idx)
    out["median_reversion_channel_atr"]        = pd.Series(_lag1_float(atr),        index=idx)
    out["median_reversion_channel_pct_pos"]    = pd.Series(_lag1_float(pct_pos),    index=idx)
    out["median_reversion_channel_lower_touch"]= pd.Series(_lag1(lower_touch),      index=idx, dtype=bool)
    out["median_reversion_channel_upper_touch"]= pd.Series(_lag1(upper_touch),      index=idx, dtype=bool)

    return out