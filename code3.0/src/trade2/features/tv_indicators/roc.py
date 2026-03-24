"""
features/roc.py - Rate of Change momentum indicator.
Computes percentage price change over lookback period.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_roc_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Rate of Change momentum features.

    Signals:
      roc_bull: ROC > 0 (positive momentum)
      roc_bear: ROC < 0 (negative momentum)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["roc"].

    Returns:
        df copy with ROC columns added.
    """
    roc_cfg = config.get("tv_indicators", {}).get("roc", {})
    period = roc_cfg.get("period", 12)

    out = df.copy()
    close = out["Close"].astype(float).values

    # Calculate ROC: ((current - n_periods_ago) / n_periods_ago) * 100
    roc = talib.ROC(close, timeperiod=period)

    # Boolean signals
    bull_raw = roc > 0
    bear_raw = roc < 0

    # Shift(1) for lag safety
    idx = out.index
    out["roc_value"] = pd.Series(roc, index=idx).shift(1)
    out["roc_bull"] = pd.Series(bull_raw, index=idx).shift(1).astype(bool)
    out["roc_bear"] = pd.Series(bear_raw, index=idx).shift(1).astype(bool)

    return out