"""
features/donchian_breakout_momentum.py - Donchian Channel Breakout with Volume Surge confirmation.
Computes breakout signals from Donchian channels confirmed by volume surge above rolling average.
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
      - Donchian Channel: highest high and lowest low over `period` bars
      - Volume surge: current volume > rolling mean volume * vol_mult
      - Bull breakout: Close breaks above upper Donchian band AND volume surge
      - Bear breakout: Close breaks below lower Donchian band AND volume surge

    Signals:
      donchian_breakout_momentum_bull: True when bullish breakout (close > upper band + vol surge)
      donchian_breakout_momentum_bear: True when bearish breakout (close < lower band + vol surge)

    Args:
        df:     OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        config: Full config dict. Reads config["tv_indicators"]["donchian_breakout_momentum"].

    Returns:
        df copy with Donchian Breakout Momentum columns added (all shift(1)).
    """
    params = config.get("tv_indicators", {}).get("donchian_breakout_momentum", {})
    period     = int(params.get("period", 8))
    vol_mult   = float(params.get("vol_mult", 1.5))
    vol_period = int(params.get("vol_period", 6))

    # Clamp to scalping-safe range
    period     = max(3, min(period, 14))
    vol_period = max(3, min(vol_period, 14))

    out    = df.copy()
    close  = out["Close"].astype(float).values
    high   = out["High"].astype(float).values
    low    = out["Low"].astype(float).values
    volume = out["Volume"].astype(float).values
    n      = len(close)

    # --- Donchian Channel ---
    # Upper band: highest high over `period` bars
    upper = talib.MAX(high, timeperiod=period)
    # Lower band: lowest low over `period` bars
    lower = talib.MIN(low,  timeperiod=period)
    # Mid band: midpoint of channel
    mid   = (upper + lower) / 2.0

    # --- Volume Surge ---
    # Rolling mean volume over vol_period bars using talib SMA
    vol_ma  = talib.SMA(volume, timeperiod=vol_period)
    vol_surge = volume > (vol_ma * vol_mult)

    # --- Breakout Detection (raw, before shift) ---
    # Bull: close breaks ABOVE upper Donchian band with volume surge
    bull_raw = (close > upper) & vol_surge

    # Bear: close breaks BELOW lower Donchian band with volume surge
    bear_raw = (close < lower) & vol_surge

    # --- Momentum Confirmation via Rate-of-Change ---
    # Use short ROC to confirm directional momentum (period=3 for scalping)
    roc_period = max(3, min(period // 2, 5))
    roc = talib.ROC(close, timeperiod=roc_period)

    # Filter: bull only when momentum positive, bear only when momentum negative
    bull_confirmed = bull_raw & (roc > 0.0)
    bear_confirmed = bear_raw & (roc < 0.0)

    # --- Channel Width (normalized) ---
    channel_width = np.where(mid != 0.0, (upper - lower) / mid, 0.0)

    # --- Helper: numpy 1-bar lag ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        result = np.empty_like(arr)
        result[0] = arr.dtype.type(0)  # False for bool, 0.0 for float
        result[1:] = arr[:-1]
        return result

    idx = out.index

    # --- All outputs shifted by 1 bar for lag safety ---
    out["donchian_breakout_momentum_upper"]         = pd.Series(_lag1(upper),            index=idx)
    out["donchian_breakout_momentum_lower"]         = pd.Series(_lag1(lower),            index=idx)
    out["donchian_breakout_momentum_mid"]           = pd.Series(_lag1(mid),              index=idx)
    out["donchian_breakout_momentum_channel_width"] = pd.Series(_lag1(channel_width),    index=idx)
    out["donchian_breakout_momentum_vol_ma"]        = pd.Series(_lag1(vol_ma),           index=idx)
    out["donchian_breakout_momentum_vol_surge"]     = pd.Series(_lag1(vol_surge),        index=idx, dtype=bool)
    out["donchian_breakout_momentum_roc"]           = pd.Series(_lag1(roc),              index=idx)
    out["donchian_breakout_momentum_bull"]          = pd.Series(_lag1(bull_confirmed),   index=idx, dtype=bool)
    out["donchian_breakout_momentum_bear"]          = pd.Series(_lag1(bear_confirmed),   index=idx, dtype=bool)

    return out