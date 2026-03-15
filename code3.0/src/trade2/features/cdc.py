"""
features/cdc.py - CDC Action Zone feature detection.
Computes EMA-of-OHLC4 based trend zones (green/yellow/blue/red).
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_cdc_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute CDC Action Zone features.

    Zones:
      green:  Fast > Slow AND Close > Fast   (strong bull)
      yellow: Fast > Slow AND Close <= Fast  (weak bull)
      blue:   Fast < Slow AND Close < Fast   (strong bear)
      red:    Fast < Slow AND Close >= Fast  (weak bear)

    Signals:
      cdc_buy:  first bar transitioning into green zone (shift(1))
      cdc_sell: first bar transitioning into blue zone (shift(1))

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["strategies"]["cdc"].

    Returns:
        df copy with CDC columns added.
    """
    cdc_cfg = config["strategies"]["cdc"]
    ap_period   = cdc_cfg["ap_period"]
    fast_period = cdc_cfg["fast_period"]
    slow_period = cdc_cfg["slow_period"]

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values
    open_ = out["Open"].astype(float).values

    # OHLC4 source
    src = (open_ + high + low + close) / 4.0

    # EMA of source (AP), then Fast/Slow EMAs of AP
    ap   = talib.EMA(src,  timeperiod=ap_period)
    fast = talib.EMA(ap,   timeperiod=fast_period)
    slow = talib.EMA(ap,   timeperiod=slow_period)

    # Zone classification (raw, before shift)
    bull_cross = fast > slow
    zone_green  = bull_cross & (close > fast)
    zone_yellow = bull_cross & (close <= fast)
    zone_blue   = ~bull_cross & (close < fast)
    zone_red    = ~bull_cross & (close >= fast)

    # Helper: numpy-level 1-bar lag (avoids pandas object-dtype FutureWarning)
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    # Transition signals: entering green or blue (first bar of transition)
    cdc_buy_raw  = zone_green & ~_lag1(zone_green)
    cdc_sell_raw = zone_blue  & ~_lag1(zone_blue)

    # Shift(1) for lag safety -- signals usable only on next bar
    idx = out.index
    out["cdc_zone_green"]  = pd.Series(_lag1(zone_green),    index=idx, dtype=bool)
    out["cdc_zone_yellow"] = pd.Series(_lag1(zone_yellow),   index=idx, dtype=bool)
    out["cdc_zone_blue"]   = pd.Series(_lag1(zone_blue),     index=idx, dtype=bool)
    out["cdc_zone_red"]    = pd.Series(_lag1(zone_red),      index=idx, dtype=bool)
    out["cdc_buy"]         = pd.Series(_lag1(cdc_buy_raw),   index=idx, dtype=bool)
    out["cdc_sell"]        = pd.Series(_lag1(cdc_sell_raw),  index=idx, dtype=bool)
    out["cdc_fast"]        = pd.Series(fast, index=out.index).shift(1)
    out["cdc_slow"]        = pd.Series(slow, index=out.index).shift(1)

    return out
