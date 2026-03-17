"""
features/keltner_channel.py - Keltner Channel feature detection.
Computes EMA-based channel with ATR bands and price position.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_keltner_channel_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Keltner Channel features.

    Channel components:
      middle: EMA of HLC3 (period)
      atr: ATR(period)
      upper: middle + atr_mult * atr
      lower: middle - atr_mult * atr

    Features:
      Channel boundaries (shifted)
      Position of close relative to channel (0-1 normalized)
      Breakout/down signals (close outside bands)

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["keltner_channel"].

    Returns:
        df copy with keltner_channel_* columns added.
    """
    # Get parameters
    kc_config = config.get('tv_indicators', {}).get('keltner_channel', {})
    period = kc_config.get('period', 20)
    atr_mult = kc_config.get('atr_mult', 2.0)

    out = df.copy()
    high = out["High"].astype(float).values
    low = out["Low"].astype(float).values
    close = out["Close"].astype(float).values

    # Calculate HLC3
    hlc3 = (high + low + close) / 3.0

    # Middle line (EMA of HLC3)
    middle = talib.EMA(hlc3, timeperiod=period)

    # ATR for bands
    atr = talib.ATR(high, low, close, timeperiod=period)

    # Upper and lower bands
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr

    # Calculate position within channel (0-1 normalized)
    # Avoid division by zero
    channel_width = upper - lower
    position = (close - lower) / np.where(channel_width != 0, channel_width, 1.0)

    # Breakout signals
    above_upper = close > upper
    below_lower = close < lower

    # Shift all outputs by 1 for lag safety
    out["keltner_channel_middle"] = pd.Series(middle, index=out.index).shift(1)
    out["keltner_channel_upper"] = pd.Series(upper, index=out.index).shift(1)
    out["keltner_channel_lower"] = pd.Series(lower, index=out.index).shift(1)
    out["keltner_channel_atr"] = pd.Series(atr, index=out.index).shift(1)
    out["keltner_channel_position"] = pd.Series(position, index=out.index).shift(1)
    out["keltner_channel_above_upper"] = pd.Series(above_upper, index=out.index).shift(1)
    out["keltner_channel_below_lower"] = pd.Series(below_lower, index=out.index).shift(1)

    return out