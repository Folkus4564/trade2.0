"""
features/smc.py - Smart Money Concepts feature detection.
All params explicit (no config globals). All signals shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from trade2.features.indicators import compute_atr_pandas


def add_smc_features(
    df: pd.DataFrame,
    ob_validity: int = 20,
    fvg_validity: int = 15,
    swing_lookback: int = 20,
    ob_impulse_bars: int = 3,
    ob_impulse_mult: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Add SMC features: Order Blocks, Fair Value Gaps, Liquidity Sweeps.
    All signals are shift(1) so they represent completed-bar conditions.

    Args:
        ob_validity:     Bars an Order Block stays valid
        fvg_validity:    Bars a Fair Value Gap stays valid
        swing_lookback:  Lookback bars for swing high/low detection
        ob_impulse_bars: Bars in impulse window
        ob_impulse_mult: ATR multiplier for impulse threshold
        atr_period:      ATR lookback period

    Returns:
        df copy with ob_bullish, ob_bearish, fvg_bullish, fvg_bearish,
        sweep_low, sweep_high columns added.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    open_ = out["Open"].astype(float)

    n = len(out)
    atr = compute_atr_pandas(high, low, close, atr_period)
    atr_arr   = atr.values
    close_arr = close.values
    high_arr  = high.values
    low_arr   = low.values
    open_arr  = open_.values

    # ---- Order Blocks ----
    bull_ob_low  = np.full(n, np.nan)
    bull_ob_high = np.full(n, np.nan)
    bear_ob_low  = np.full(n, np.nan)
    bear_ob_high = np.full(n, np.nan)

    for i in range(ob_impulse_bars + 1, n):
        impulse_up   = close_arr[i-1] - min(low_arr[i-ob_impulse_bars:i])
        impulse_down = max(high_arr[i-ob_impulse_bars:i]) - close_arr[i-1]
        threshold    = ob_impulse_mult * atr_arr[i-1] if not np.isnan(atr_arr[i-1]) else 0.0

        if threshold <= 0:
            continue

        if impulse_up >= threshold:
            for j in range(i-1, max(i-ob_impulse_bars-1, -1), -1):
                if open_arr[j] > close_arr[j]:  # bearish candle = bullish OB
                    if (i - j) <= ob_validity:
                        bull_ob_low[i]  = low_arr[j]
                        bull_ob_high[i] = high_arr[j]
                    break

        if impulse_down >= threshold:
            for j in range(i-1, max(i-ob_impulse_bars-1, -1), -1):
                if close_arr[j] > open_arr[j]:  # bullish candle = bearish OB
                    if (i - j) <= ob_validity:
                        bear_ob_low[i]  = low_arr[j]
                        bear_ob_high[i] = high_arr[j]
                    break

    bull_ob_retest = (
        (~np.isnan(bull_ob_low)) &
        (close_arr >= bull_ob_low) &
        (close_arr <= bull_ob_high)
    )
    bear_ob_retest = (
        (~np.isnan(bear_ob_low)) &
        (close_arr >= bear_ob_low) &
        (close_arr <= bear_ob_high)
    )

    out["ob_bullish"] = pd.Series(bull_ob_retest, index=out.index, dtype=bool).shift(1).fillna(False)
    out["ob_bearish"] = pd.Series(bear_ob_retest, index=out.index, dtype=bool).shift(1).fillna(False)

    # ---- Fair Value Gaps ----
    bull_fvg_lo = np.full(n, np.nan)
    bull_fvg_hi = np.full(n, np.nan)
    bear_fvg_lo = np.full(n, np.nan)
    bear_fvg_hi = np.full(n, np.nan)

    a_bull_lo, a_bull_hi, a_bull_age = np.nan, np.nan, 0
    a_bear_lo, a_bear_hi, a_bear_age = np.nan, np.nan, 0

    for i in range(2, n):
        h_im2 = high_arr[i-2]
        l_im2 = low_arr[i-2]
        l_i   = low_arr[i]
        h_i   = high_arr[i]

        if h_im2 < l_i:
            a_bull_lo, a_bull_hi, a_bull_age = h_im2, l_i, 0
        if l_im2 > h_i:
            a_bear_lo, a_bear_hi, a_bear_age = h_i, l_im2, 0

        if not np.isnan(a_bull_lo):
            a_bull_age += 1
            if close_arr[i] < a_bull_lo or a_bull_age > fvg_validity:
                a_bull_lo, a_bull_hi, a_bull_age = np.nan, np.nan, 0
            else:
                bull_fvg_lo[i] = a_bull_lo
                bull_fvg_hi[i] = a_bull_hi

        if not np.isnan(a_bear_lo):
            a_bear_age += 1
            if close_arr[i] > a_bear_hi or a_bear_age > fvg_validity:
                a_bear_lo, a_bear_hi, a_bear_age = np.nan, np.nan, 0
            else:
                bear_fvg_lo[i] = a_bear_lo
                bear_fvg_hi[i] = a_bear_hi

    bull_fvg_retest = (
        (~np.isnan(bull_fvg_lo)) &
        (close_arr >= bull_fvg_lo) &
        (close_arr <= bull_fvg_hi)
    )
    bear_fvg_retest = (
        (~np.isnan(bear_fvg_lo)) &
        (close_arr >= bear_fvg_lo) &
        (close_arr <= bear_fvg_hi)
    )

    out["fvg_bullish"] = pd.Series(bull_fvg_retest, index=out.index, dtype=bool).shift(1).fillna(False)
    out["fvg_bearish"] = pd.Series(bear_fvg_retest, index=out.index, dtype=bool).shift(1).fillna(False)

    # ---- Liquidity Sweeps ----
    swing_high = high.rolling(swing_lookback).max().shift(1)
    swing_low  = low.rolling(swing_lookback).min().shift(1)

    sweep_low_raw  = (low < swing_low)  & (close > swing_low)  & (close > open_)
    sweep_high_raw = (high > swing_high) & (close < swing_high) & (close < open_)

    out["sweep_low"]  = sweep_low_raw.shift(1).fillna(False)
    out["sweep_high"] = sweep_high_raw.shift(1).fillna(False)

    return out


def add_pin_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rejection/pin bar candles. Shifted by 1 for lag safety.
    Bullish pin bar: long lower wick, close > open.
    Bearish pin bar: long upper wick, close < open.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    open_ = out["Open"].astype(float)

    body       = (close - open_).abs()
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

    pin_bull = (lower_wick > 2 * body) & (lower_wick > upper_wick) & (close > open_)
    pin_bear = (upper_wick > 2 * body) & (upper_wick > lower_wick) & (close < open_)

    out["pin_bar_bull"] = pin_bull.shift(1).fillna(False).astype(bool)
    out["pin_bar_bear"] = pin_bear.shift(1).fillna(False).astype(bool)

    return out
