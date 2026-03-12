"""
backtesting/costs.py - Cost computation utilities.
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_slippage_array(
    close: pd.Series,
    config: Dict[str, Any],
) -> pd.Series:
    """
    Compute per-bar slippage fraction from spread + slippage config.
    Applies Asian-hours spread multiplier and high-volatility multiplier per bar.

    Vol multiplier is activated when the bar's range proxy (|close diff|) exceeds
    1.5x the rolling median over spread_vol_atr_lookback bars.

    Returns:
        pd.Series of slippage fractions, one per bar (same index as close).
    """
    costs_cfg    = config["costs"]
    spread_pips  = costs_cfg["spread_pips"]
    slip_pips    = costs_cfg["slippage_pips"]
    asian_mult   = costs_cfg["spread_asian_mult"]
    vol_mult     = costs_cfg["spread_vol_mult"]
    atr_lb       = costs_cfg["spread_vol_atr_lookback"]
    pip_value    = 0.01  # XAUUSD: 1 pip = $0.01

    _ASIAN_HOURS = set(range(0, 7)) | set(range(22, 24))

    avg_price = float(close.mean())

    # Per-bar Asian-hour flag
    if hasattr(close.index, "hour"):
        idx_hours = close.index.hour if close.index.tz is None else close.index.tz_convert("UTC").hour
        is_asian  = pd.Series(idx_hours, index=close.index).isin(_ASIAN_HOURS)
    else:
        is_asian  = pd.Series(False, index=close.index)

    # Per-bar high-volatility flag: activate when |close diff| > 1.5x rolling median
    vol_proxy       = close.diff().abs()
    rolling_median  = vol_proxy.rolling(atr_lb, min_periods=1).median()
    is_hvol         = vol_proxy > (rolling_median * 1.5)

    spread_base = float(spread_pips)
    spread_arr  = pd.Series(spread_base, index=close.index)
    spread_arr  = np.where(
        is_asian & is_hvol, spread_arr * asian_mult * vol_mult,
        np.where(is_asian,  spread_arr * asian_mult,
        np.where(is_hvol,   spread_arr * vol_mult,
                            spread_arr))
    )
    slippage_arr = ((pd.Series(spread_arr, index=close.index) + slip_pips) * pip_value) / (avg_price + 1e-10)
    return slippage_arr


def compute_slippage(
    close: pd.Series,
    config: Dict[str, Any],
) -> float:
    """
    Compute blended average slippage fraction (scalar).
    Backward-compatible wrapper around compute_slippage_array().
    Used by doubled_costs sensitivity tests and any caller expecting a scalar.
    """
    return float(compute_slippage_array(close, config).mean())


def doubled_costs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of config with all costs doubled (for 2x cost sensitivity test).
    """
    cfg = copy.deepcopy(config)
    costs = cfg["costs"]
    costs["spread_pips"]   = costs["spread_pips"]   * 2
    costs["slippage_pips"] = costs["slippage_pips"] * 2
    costs["commission_rt"] = costs["commission_rt"] * 2
    cfg["costs"] = costs
    return cfg
