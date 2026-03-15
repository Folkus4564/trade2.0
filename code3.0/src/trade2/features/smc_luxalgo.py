"""
features/smc_luxalgo.py - LuxAlgo-style SMC feature detection.

Implements fractal swing detection, BOS/CHoCH, premium/discount zones,
and equal highs/lows. All outputs shift(1) for lag safety.

Design note: swing detection uses right_bars future bars for confirmation
(standard fractal definition). When computed in batch mode on a full DataFrame
this is fine; the shift(1) ensures no future data leaks into entry decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _lag1(arr: np.ndarray) -> np.ndarray:
    """Shift boolean numpy array right by 1, filling position 0 with False."""
    out = np.empty_like(arr)
    out[0] = False
    out[1:] = arr[:-1]
    return out


def detect_swing_points(
    df: pd.DataFrame,
    left_bars: int,
    right_bars: int,
) -> pd.DataFrame:
    """
    Detect fractal swing highs and lows.

    A swing high at bar i: high[i] is the maximum over the window
    [i - left_bars, i + right_bars].
    A swing low at bar i: low[i] is the minimum over the same window.

    Results are shift(1) so they are available on the bar AFTER confirmation.

    Args:
        df:         OHLCV DataFrame.
        left_bars:  Left wing of fractal.
        right_bars: Right wing of fractal (confirmation lag).

    Returns:
        df copy with swing_high (bool), swing_low (bool),
        swing_high_price (float/NaN), swing_low_price (float/NaN).
    """
    out  = df.copy()
    high = out["High"].astype(float)
    low  = out["Low"].astype(float)

    window = left_bars + right_bars + 1

    # Rolling max/min centered: high[i] is swing high iff it equals the
    # max of the full window centered on i.
    # Use center=True in rolling -- this creates lookahead by right_bars bars.
    # We compensate with shift(right_bars + 1) at the end to get full lag safety.
    roll_max = high.rolling(window=window, center=True, min_periods=window).max()
    roll_min = low.rolling(window=window,  center=True, min_periods=window).min()

    swing_high_raw = (high == roll_max)
    swing_low_raw  = (low  == roll_min)

    # Consecutive equal values can both fire -- keep only first of each run
    sh_arr = swing_high_raw.values.astype(bool)
    sl_arr = swing_low_raw.values.astype(bool)
    sh_arr = sh_arr & ~_lag1(sh_arr)
    sl_arr = sl_arr & ~_lag1(sl_arr)

    swing_high_price_raw = high.values.copy()
    swing_high_price_raw[~sh_arr] = np.nan
    swing_low_price_raw  = low.values.copy()
    swing_low_price_raw[~sl_arr]  = np.nan

    # Shift by (right_bars + 1): right_bars for confirmation, +1 for lag safety
    shift_n = right_bars + 1
    idx = out.index
    out["swing_high"]       = pd.Series(np.concatenate([np.zeros(shift_n, dtype=bool), sh_arr[:-shift_n]]),               index=idx, dtype=bool)
    out["swing_low"]        = pd.Series(np.concatenate([np.zeros(shift_n, dtype=bool), sl_arr[:-shift_n]]),               index=idx, dtype=bool)
    out["swing_high_price"] = pd.Series(np.concatenate([np.full(shift_n, np.nan),      swing_high_price_raw[:-shift_n]]), index=idx)
    out["swing_low_price"]  = pd.Series(np.concatenate([np.full(shift_n, np.nan),      swing_low_price_raw[:-shift_n]]),  index=idx)

    return out


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    Requires swing_high, swing_low, swing_high_price, swing_low_price columns
    (from detect_swing_points).

    BOS (Break of Structure): price breaks beyond swing in SAME direction as
    current market structure.
    CHoCH (Change of Character): price breaks beyond swing AGAINST current
    structure, signalling a potential reversal.

    All outputs shift(1) for lag safety.

    Returns:
        df copy with bos_bullish, bos_bearish, choch_bullish, choch_bearish columns.
    """
    out   = df.copy()
    close = out["Close"].astype(float).values
    n     = len(out)

    swing_high_price = out["swing_high_price"].values
    swing_low_price  = out["swing_low_price"].values

    bos_bull   = np.zeros(n, dtype=bool)
    bos_bear   = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)

    last_sh = np.nan   # last confirmed swing high price
    last_sl = np.nan   # last confirmed swing low price
    structure = 0      # 1 = bullish, -1 = bearish, 0 = undefined

    for i in range(n):
        # Update last known swing levels (forward-fill)
        if not np.isnan(swing_high_price[i]):
            last_sh = swing_high_price[i]
        if not np.isnan(swing_low_price[i]):
            last_sl = swing_low_price[i]

        if np.isnan(last_sh) or np.isnan(last_sl):
            continue

        c = close[i]

        if structure == 0:
            # Initialise structure from first break
            if c > last_sh:
                structure = 1
            elif c < last_sl:
                structure = -1
        elif structure == 1:  # Bullish structure
            if c > last_sh:
                bos_bull[i] = True     # Bullish BOS: continues higher
            elif c < last_sl:
                choch_bear[i] = True   # CHoCH: structure flips bearish
                structure = -1
        elif structure == -1:  # Bearish structure
            if c < last_sl:
                bos_bear[i] = True     # Bearish BOS: continues lower
            elif c > last_sh:
                choch_bull[i] = True   # CHoCH: structure flips bullish
                structure = 1

    idx = out.index
    out["bos_bullish"]   = pd.Series(_lag1(bos_bull),   index=idx, dtype=bool)
    out["bos_bearish"]   = pd.Series(_lag1(bos_bear),   index=idx, dtype=bool)
    out["choch_bullish"] = pd.Series(_lag1(choch_bull), index=idx, dtype=bool)
    out["choch_bearish"] = pd.Series(_lag1(choch_bear), index=idx, dtype=bool)

    return out


def compute_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute premium/discount zones relative to current swing range.

    Requires swing_high_price and swing_low_price columns.

    equilibrium = (last_swing_high + last_swing_low) / 2
    pd_ratio    = (close - last_swing_low) / (last_swing_high - last_swing_low)
    in_premium  = pd_ratio > 0.5   (price above equilibrium)
    in_discount = pd_ratio < 0.5   (price below equilibrium)

    All outputs shift(1) for lag safety.

    Returns:
        df copy with in_premium (bool), in_discount (bool), pd_ratio (float).
    """
    out   = df.copy()
    close = out["Close"].astype(float)

    last_sh = out["swing_high_price"].ffill()
    last_sl = out["swing_low_price"].ffill()

    swing_range = (last_sh - last_sl).clip(lower=1e-9)
    pd_ratio    = (close - last_sl) / swing_range

    out["in_premium"]  = pd.Series(_lag1((pd_ratio > 0.5).values), index=out.index, dtype=bool)
    out["in_discount"] = pd.Series(_lag1((pd_ratio < 0.5).values), index=out.index, dtype=bool)
    out["pd_ratio"]    = pd_ratio.shift(1)

    return out


def detect_equal_highs_lows(df: pd.DataFrame, atr_mult: float) -> pd.DataFrame:
    """
    Detect equal highs and equal lows (liquidity clusters).

    Two consecutive confirmed swing highs within atr_mult * ATR of each other
    are classified as equal highs (liquidity resting above).
    Same logic for equal lows.

    All outputs shift(1) for lag safety.

    Returns:
        df copy with equal_highs (bool), equal_lows (bool).
    """
    out = df.copy()
    n   = len(out)

    atr = out["atr_14"].astype(float) if "atr_14" in out.columns else pd.Series(np.nan, index=out.index)

    swing_high_prices = out["swing_high_price"].values
    swing_low_prices  = out["swing_low_price"].values
    atr_vals          = atr.values

    equal_highs = np.zeros(n, dtype=bool)
    equal_lows  = np.zeros(n, dtype=bool)

    prev_sh = np.nan
    prev_sl = np.nan

    for i in range(n):
        thr = atr_mult * atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0

        if not np.isnan(swing_high_prices[i]):
            if not np.isnan(prev_sh) and abs(swing_high_prices[i] - prev_sh) <= thr:
                equal_highs[i] = True
            prev_sh = swing_high_prices[i]

        if not np.isnan(swing_low_prices[i]):
            if not np.isnan(prev_sl) and abs(swing_low_prices[i] - prev_sl) <= thr:
                equal_lows[i] = True
            prev_sl = swing_low_prices[i]

    idx = out.index
    out["equal_highs"] = pd.Series(_lag1(equal_highs), index=idx, dtype=bool)
    out["equal_lows"]  = pd.Series(_lag1(equal_lows),  index=idx, dtype=bool)

    return out


def add_luxalgo_smc_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    config_key: str = "smc_luxalgo",
) -> pd.DataFrame:
    """
    Orchestrate all LuxAlgo SMC feature computation.

    Args:
        df:         OHLCV DataFrame (already has atr_14 column).
        config:     Full config dict.
        config_key: Key to read from config ("smc_luxalgo" or "smc_luxalgo_5m").

    Returns:
        df with all LuxAlgo SMC columns added. Unchanged if enabled=False.
    """
    lux_cfg = config[config_key]

    if not lux_cfg["enabled"]:
        return df

    left_bars  = lux_cfg["swing_left_bars"]
    right_bars = lux_cfg["swing_right_bars"]
    ehl_mult   = lux_cfg["equal_hl_atr_mult"]

    out = detect_swing_points(df, left_bars, right_bars)
    out = detect_bos_choch(out)
    out = compute_premium_discount(out)
    out = detect_equal_highs_lows(out, ehl_mult)

    return out
