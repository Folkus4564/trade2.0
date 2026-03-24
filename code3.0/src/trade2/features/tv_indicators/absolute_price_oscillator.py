"""
features/absolute_price_oscillator.py - Absolute Price Oscillator feature detection.
Computes fast momentum oscillator for scalping on 5M charts.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_absolute_price_oscillator_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Absolute Price Oscillator (APO) for 5M scalping.

    APO = Price - EMA(Price, period)
    Bullish when APO > 0 and rising (momentum positive)
    Bearish when APO < 0 and falling (momentum negative)

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["absolute_price_oscillator"].

    Returns:
        df copy with APO columns added.
    """
    indicator_cfg = config.get("tv_indicators", {}).get("absolute_price_oscillator", {})
    period = indicator_cfg.get("period", 8)
    price_source = indicator_cfg.get("price_source", "hlc3")

    out = df.copy()

    # Compute price source
    if price_source == "hlc3":
        price = (out["High"].astype(float) + out["Low"].astype(float) + out["Close"].astype(float)) / 3.0
    elif price_source == "close":
        price = out["Close"].astype(float)
    elif price_source == "hl2":
        price = (out["High"].astype(float) + out["Low"].astype(float)) / 2.0
    elif price_source == "ohlc4":
        price = (out["Open"].astype(float) + out["High"].astype(float) + 
                 out["Low"].astype(float) + out["Close"].astype(float)) / 4.0
    else:
        price = out["Close"].astype(float)

    # Compute EMA and APO
    price_vals = price.values
    ema = talib.EMA(price_vals, timeperiod=period)
    apo = price_vals - ema

    # Bullish: APO > 0 AND APO rising (current > previous)
    apo_prev = np.roll(apo, 1)
    apo_prev[0] = apo[0]  # Handle first element
    bullish_raw = (apo > 0) & (apo > apo_prev)

    # Bearish: APO < 0 AND APO falling (current < previous)
    bearish_raw = (apo < 0) & (apo < apo_prev)

    # Apply shift(1) for lag safety
    out["absolute_price_oscillator_apo"] = pd.Series(apo, index=out.index).shift(1)
    out["absolute_price_oscillator_bull"] = pd.Series(bullish_raw, index=out.index).shift(1)
    out["absolute_price_oscillator_bear"] = pd.Series(bearish_raw, index=out.index).shift(1)

    return out