"""
features/vortex.py - Vortex Indicator feature detection.
Computes VI+ (positive trend movement) and VI- (negative trend movement).
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_vortex_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Vortex Indicator (VI+ and VI-).
    
    Formula:
      TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
      VM+ = abs(high - prev_low)
      VM- = abs(low - prev_high)
      
      VI+ = SUM(VM+, period) / SUM(TR, period)
      VI- = SUM(VM-, period) / SUM(TR, period)
    
    Args:
        df: OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["vortex"].
    
    Returns:
        df copy with Vortex columns added.
    """
    vortex_cfg = config.get('tv_indicators', {}).get('vortex', {})
    period = vortex_cfg.get('period', 14)
    
    out = df.copy()
    high = out['High'].astype(float).values
    low = out['Low'].astype(float).values
    close = out['Close'].astype(float).values
    
    # True Range (TR)
    tr = talib.TRANGE(high, low, close)
    
    # Vortex Movement components (VM+ and VM-)
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]  # First bar: use current value
    
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    
    vm_plus = np.abs(high - prev_low)
    vm_minus = np.abs(low - prev_high)
    
    # Sum over period using rolling window
    tr_sum = pd.Series(tr).rolling(window=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period).sum().values
    
    # Vortex Indicator values (VI+ and VI-)
    vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
    
    # Shift(1) for lag safety
    idx = out.index
    out['vortex_vi_plus'] = pd.Series(vi_plus, index=idx).shift(1)
    out['vortex_vi_minus'] = pd.Series(vi_minus, index=idx).shift(1)
    
    return out