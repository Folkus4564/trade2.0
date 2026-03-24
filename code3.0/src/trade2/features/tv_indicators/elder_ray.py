"""
elder_ray.py - Elder Ray Bull/Bear Power feature detection.
Computes High/Low minus EMA(Close) to measure buying/selling force.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_elder_ray_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Elder Ray Bull/Bear Power features.

    Bull Power = High - EMA(Close, period)
    Bear Power = Low - EMA(Close, period)

    Signals:
      elder_ray_bull: Bull Power > 0 (bullish momentum)
      elder_ray_bear: Bear Power < 0 (bearish momentum)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["elder_ray"].

    Returns:
        df copy with Elder Ray columns added.
    """
    cfg = config.get("tv_indicators", {}).get("elder_ray", {})
    period = cfg.get("period", 13)

    out = df.copy()
    close = out["Close"].astype(float).values
    high = out["High"].astype(float).values
    low = out["Low"].astype(float).values

    # EMA of Close
    ema = talib.EMA(close, timeperiod=period)

    # Bull and Bear Power
    bull_power = high - ema
    bear_power = low - ema

    # Boolean signals
    bull_raw = bull_power > 0
    bear_raw = bear_power < 0

    # Apply shift(1) for lag safety
    idx = out.index
    out["elder_ray_bull"] = pd.Series(bull_raw, index=idx).shift(1).fillna(False).astype(bool)
    out["elder_ray_bear"] = pd.Series(bear_raw, index=idx).shift(1).fillna(False).astype(bool)
    out["elder_ray_bull_power"] = pd.Series(bull_power, index=idx).shift(1)
    out["elder_ray_bear_power"] = pd.Series(bear_power, index=idx).shift(1)
    out["elder_ray_ema"] = pd.Series(ema, index=idx).shift(1)

    return out