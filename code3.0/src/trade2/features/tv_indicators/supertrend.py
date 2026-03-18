"""
features/supertrend.py - Supertrend indicator.
ATR-based trailing stop that flips direction on breakout.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_supertrend_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Supertrend indicator features.

    Bullish when price is above supertrend (green), 
    Bearish when price is below supertrend (red).

    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["supertrend"].

    Returns:
        df copy with supertrend columns added.
    """
    supertrend_cfg = config.get("tv_indicators", {}).get("supertrend", {})
    period = supertrend_cfg.get("period", 10)
    multiplier = supertrend_cfg.get("multiplier", 3.0)

    out = df.copy()
    high = out["High"].astype(float).values
    low = out["Low"].astype(float).values
    close = out["Close"].astype(float).values

    # Calculate ATR
    atr = talib.ATR(high, low, close, timeperiod=period)

    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize arrays
    final_upper_band = np.full_like(close, np.nan)
    final_lower_band = np.full_like(close, np.nan)
    supertrend = np.full_like(close, np.nan)
    direction = np.zeros_like(close)  # 1 for bullish, -1 for bearish

    # First valid index
    start_idx = period

    if len(close) > start_idx:
        # Initialize
        final_upper_band[start_idx] = upper_band[start_idx]
        final_lower_band[start_idx] = lower_band[start_idx]
        direction[start_idx] = 1
        supertrend[start_idx] = final_lower_band[start_idx]

        # Iterate to adjust bands and determine direction
        for i in range(start_idx + 1, len(close)):
            # Adjust upper band
            if upper_band[i] < final_upper_band[i - 1] or close[i - 1] > final_upper_band[i - 1]:
                final_upper_band[i] = upper_band[i]
            else:
                final_upper_band[i] = final_upper_band[i - 1]

            # Adjust lower band
            if lower_band[i] > final_lower_band[i - 1] or close[i - 1] < final_lower_band[i - 1]:
                final_lower_band[i] = lower_band[i]
            else:
                final_lower_band[i] = final_lower_band[i - 1]

            # Determine direction
            if direction[i - 1] == 1:  # Previous was bullish
                if close[i] <= final_lower_band[i]:  # Flip to bearish
                    direction[i] = -1
                else:  # Stay bullish
                    direction[i] = 1
            else:  # Previous was bearish
                if close[i] >= final_upper_band[i]:  # Flip to bullish
                    direction[i] = 1
                else:  # Stay bearish
                    direction[i] = -1

            # Set supertrend value
            if direction[i] == 1:
                supertrend[i] = final_lower_band[i]
            else:
                supertrend[i] = final_upper_band[i]

    # Create boolean series (shifted by 1)
    idx = out.index
    out["supertrend_bull"] = pd.Series(direction == 1, index=idx).shift(1)
    out["supertrend_bear"] = pd.Series(direction == -1, index=idx).shift(1)
    out["supertrend_value"] = pd.Series(supertrend, index=idx).shift(1)
    out["supertrend_upper"] = pd.Series(final_upper_band, index=idx).shift(1)
    out["supertrend_lower"] = pd.Series(final_lower_band, index=idx).shift(1)

    return out