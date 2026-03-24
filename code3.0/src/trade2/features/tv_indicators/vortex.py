"""
features/vortex.py - Vortex Indicator feature detection.
Computes VI+ and VI- trend lines and derived bull/bear states.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_vortex_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Vortex Indicator (VI) features.

    Formula:
      TR = MAX(High - Low, ABS(High - Close_prev), ABS(Low - Close_prev))
      VM+ = ABS(High - Low_prev)
      VM- = ABS(Low - High_prev)
      VI+ = SUM(VM+, period) / SUM(TR, period)
      VI- = SUM(VM-, period) / SUM(TR, period)

    Signals:
      vortex_bull: VI+ > VI- (trend up)
      vortex_bear: VI- > VI+ (trend down)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["vortex"].

    Returns:
        df copy with vortex columns added.
    """
    vortex_cfg = config.get("tv_indicators", {}).get("vortex", {})
    period = vortex_cfg.get("period", 14)

    out = df.copy()
    high = out["High"].astype(float).values
    low = out["Low"].astype(float).values
    close = out["Close"].astype(float).values

    # True Range
    tr = talib.TRANGE(high, low, close)

    # VM+ and VM- calculations
    high_series = pd.Series(high)
    low_series = pd.Series(low)

    vm_plus = np.abs(high - low_series.shift(1).values)
    vm_minus = np.abs(low - high_series.shift(1).values)

    # Rolling sums
    sum_tr = pd.Series(tr).rolling(window=period).sum()
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period).sum()
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period).sum()

    # VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    # Bull/bear states
    vortex_bull_raw = vi_plus > vi_minus
    vortex_bear_raw = vi_minus > vi_plus

    # Shift(1) for lag safety
    idx = out.index
    out["vortex_vi_plus"] = vi_plus.shift(1)
    out["vortex_vi_minus"] = vi_minus.shift(1)
    out["vortex_bull"] = pd.Series(vortex_bull_raw, index=idx).shift(1).fillna(False).astype(bool)
    out["vortex_bear"] = pd.Series(vortex_bear_raw, index=idx).shift(1).fillna(False).astype(bool)

    return out