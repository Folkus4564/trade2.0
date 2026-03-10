"""
Module: features.py
Purpose: Multi-timeframe feature engineering
         - 1H features: HMM regime inputs (trend, vol, momentum)
         - 5M features: SMC entry signals (OB, FVG, sweeps)
         All features are lag-safe (shift(1) before use)
"""

import numpy as np
import pandas as pd
import talib
from typing import Tuple


# =============================================================================
#  1H FEATURES (HMM regime detection)
# =============================================================================

def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half   = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = talib.WMA(series.values.astype(float), timeperiod=half)
    wma_full = talib.WMA(series.values.astype(float), timeperiod=period)
    raw      = 2 * wma_half - wma_full
    result   = talib.WMA(raw, timeperiod=sqrt_p)
    return pd.Series(result, index=series.index, name=f"HMA_{period}")


def add_1h_features(
    df: pd.DataFrame,
    hma_period: int = 55,
    ema_period: int = 21,
    atr_period: int = 14,
    rsi_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    Add 1H features for HMM regime detection.
    All HMM input features are shift(1) to prevent lookahead.

    Returns df with added columns including hmm_feat_* columns.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)

    # -- Returns --
    out["log_ret"] = np.log(close / close.shift(1))
    out["ret_5"]   = close.pct_change(5)
    out["ret_20"]  = close.pct_change(20)

    # -- Trend --
    out["hma"]       = hma(close, hma_period)
    out["hma_slope"] = out["hma"].diff(3) / (out["hma"].shift(3) + 1e-10)
    out["ema"]       = pd.Series(
        talib.EMA(close.values, timeperiod=ema_period), index=out.index
    )

    # -- Volatility --
    out["atr_14"]   = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period),
        index=out.index,
    )
    out["atr_norm"] = out["atr_14"] / (close + 1e-10)
    out["vol_20"]   = out["log_ret"].rolling(20).std() * np.sqrt(252 * 24)

    # -- Momentum --
    out["rsi_14"]   = pd.Series(
        talib.RSI(close.values, timeperiod=rsi_period), index=out.index
    )
    out["rsi_norm"] = (out["rsi_14"] - 50.0) / 50.0

    # -- Trend strength --
    out["adx_14"]   = pd.Series(
        talib.ADX(high.values, low.values, close.values, timeperiod=adx_period),
        index=out.index,
    )

    # -- MACD --
    macd_v, sig_v, _ = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    out["macd"]      = pd.Series(macd_v, index=out.index)
    out["macd_sig"]  = pd.Series(sig_v, index=out.index)
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    # -- Bollinger Bands --
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close.values, timeperiod=20)
    out["bb_width"] = pd.Series(
        (bb_upper - bb_lower) / (bb_mid + 1e-10), index=out.index
    )

    # -- HMM input features (ALL shifted 1 bar) --
    out["hmm_feat_ret"]       = out["log_ret"].shift(1)
    out["hmm_feat_rsi"]       = out["rsi_norm"].shift(1)
    out["hmm_feat_atr"]       = out["atr_norm"].shift(1)
    out["hmm_feat_vol"]       = out["vol_20"].shift(1)
    out["hmm_feat_hma_slope"] = out["hma_slope"].shift(1)
    out["hmm_feat_bb_width"]  = out["bb_width"].shift(1)
    out["hmm_feat_macd"]      = out["macd_hist"].shift(1)

    return out


def get_hmm_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
    """Extract HMM feature matrix (NaN-free rows) and index."""
    cols = [
        "hmm_feat_ret",
        "hmm_feat_rsi",
        "hmm_feat_atr",
        "hmm_feat_vol",
        "hmm_feat_hma_slope",
        "hmm_feat_bb_width",
        "hmm_feat_macd",
    ]
    feat = df[cols].dropna()
    return feat.values, feat.index


# =============================================================================
#  5M FEATURES (SMC entry signals)
# =============================================================================

def compute_atr_pandas(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """ATR using pandas (no TA-Lib dependency for SMC functions)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_5m_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    dc_period: int = 40,
    adx_period: int = 14,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Add 5M features for SMC entry detection.
    Includes: OB, FVG, liquidity sweeps, Donchian breakouts, ATR, ADX, RSI.
    All features are lag-safe.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    open_ = out["Open"].astype(float)

    # -- Basic indicators on 5M --
    out["atr_14"] = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period),
        index=out.index,
    )
    out["atr_norm"] = out["atr_14"] / (close + 1e-10)

    out["adx_14"] = pd.Series(
        talib.ADX(high.values, low.values, close.values, timeperiod=adx_period),
        index=out.index,
    )

    out["rsi_14"] = pd.Series(
        talib.RSI(close.values, timeperiod=rsi_period), index=out.index,
    )

    # -- Donchian Channel (shifted 1 bar) --
    out["dc_upper"] = high.rolling(dc_period).max().shift(1)
    out["dc_lower"] = low.rolling(dc_period).min().shift(1)
    out["dc_mid"]   = (out["dc_upper"] + out["dc_lower"]) / 2.0

    out["breakout_long"]  = (close > out["dc_upper"]).astype(int)
    out["breakout_short"] = (close < out["dc_lower"]).astype(int)

    # ATR expansion flag
    atr_ma = out["atr_14"].rolling(20).mean()
    out["atr_expansion"] = (out["atr_14"] > atr_ma).astype(int)

    # -- SMC features --
    out = _add_smc_features(out, atr_period=atr_period)

    # -- Pin bar rejection candles --
    out = _add_pin_bar_features(out)

    return out


def _add_smc_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Add SMC features to 5M DataFrame.
    Order Blocks, Fair Value Gaps, Liquidity Sweeps.
    All shifted by 1 for lag safety.
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
    OB_VALIDITY  = 60    # 60 bars on 5M = 5 hours
    IMPULSE_BARS = 3
    IMPULSE_MULT = 1.5

    bull_ob_low  = np.full(n, np.nan)
    bull_ob_high = np.full(n, np.nan)
    bear_ob_low  = np.full(n, np.nan)
    bear_ob_high = np.full(n, np.nan)

    for i in range(IMPULSE_BARS + 1, n):
        impulse_up   = close_arr[i - 1] - min(low_arr[i - IMPULSE_BARS: i])
        impulse_down = max(high_arr[i - IMPULSE_BARS: i]) - close_arr[i - 1]
        threshold    = IMPULSE_MULT * atr_arr[i - 1] if not np.isnan(atr_arr[i - 1]) else 0.0

        if threshold <= 0:
            continue

        if impulse_up >= threshold:
            for j in range(i - 1, max(i - IMPULSE_BARS - 1, -1), -1):
                if open_arr[j] > close_arr[j]:
                    if (i - j) <= OB_VALIDITY:
                        bull_ob_low[i]  = low_arr[j]
                        bull_ob_high[i] = high_arr[j]
                    break

        if impulse_down >= threshold:
            for j in range(i - 1, max(i - IMPULSE_BARS - 1, -1), -1):
                if close_arr[j] > open_arr[j]:
                    if (i - j) <= OB_VALIDITY:
                        bear_ob_low[i]  = low_arr[j]
                        bear_ob_high[i] = high_arr[j]
                    break

    bull_ob_retest_raw = (
        (~np.isnan(bull_ob_low))
        & (close_arr >= bull_ob_low)
        & (close_arr <= bull_ob_high)
    )
    bear_ob_retest_raw = (
        (~np.isnan(bear_ob_low))
        & (close_arr >= bear_ob_low)
        & (close_arr <= bear_ob_high)
    )

    out["ob_bullish"] = pd.Series(bull_ob_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)
    out["ob_bearish"] = pd.Series(bear_ob_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)

    # ---- Fair Value Gaps ----
    FVG_VALIDITY = 36  # 36 bars on 5M = 3 hours

    bull_fvg_lo = np.full(n, np.nan)
    bull_fvg_hi = np.full(n, np.nan)
    bear_fvg_lo = np.full(n, np.nan)
    bear_fvg_hi = np.full(n, np.nan)

    active_bull_fvg_lo, active_bull_fvg_hi, active_bull_fvg_age = np.nan, np.nan, 0
    active_bear_fvg_lo, active_bear_fvg_hi, active_bear_fvg_age = np.nan, np.nan, 0

    for i in range(2, n):
        h_im2 = high_arr[i - 2]
        l_im2 = low_arr[i - 2]
        l_i   = low_arr[i]
        h_i   = high_arr[i]

        if h_im2 < l_i:
            active_bull_fvg_lo, active_bull_fvg_hi, active_bull_fvg_age = h_im2, l_i, 0

        if l_im2 > h_i:
            active_bear_fvg_lo, active_bear_fvg_hi, active_bear_fvg_age = h_i, l_im2, 0

        if not np.isnan(active_bull_fvg_lo):
            active_bull_fvg_age += 1
            if close_arr[i] < active_bull_fvg_lo or active_bull_fvg_age > FVG_VALIDITY:
                active_bull_fvg_lo, active_bull_fvg_hi, active_bull_fvg_age = np.nan, np.nan, 0
            else:
                bull_fvg_lo[i] = active_bull_fvg_lo
                bull_fvg_hi[i] = active_bull_fvg_hi

        if not np.isnan(active_bear_fvg_lo):
            active_bear_fvg_age += 1
            if close_arr[i] > active_bear_fvg_hi or active_bear_fvg_age > FVG_VALIDITY:
                active_bear_fvg_lo, active_bear_fvg_hi, active_bear_fvg_age = np.nan, np.nan, 0
            else:
                bear_fvg_lo[i] = active_bear_fvg_lo
                bear_fvg_hi[i] = active_bear_fvg_hi

    bull_fvg_retest_raw = (
        (~np.isnan(bull_fvg_lo))
        & (close_arr >= bull_fvg_lo)
        & (close_arr <= bull_fvg_hi)
    )
    bear_fvg_retest_raw = (
        (~np.isnan(bear_fvg_lo))
        & (close_arr >= bear_fvg_lo)
        & (close_arr <= bear_fvg_hi)
    )

    out["fvg_bullish"] = pd.Series(bull_fvg_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)
    out["fvg_bearish"] = pd.Series(bear_fvg_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)

    # ---- Liquidity Sweeps ----
    SWING_LOOKBACK = 60  # 60 bars on 5M = 5 hours

    swing_high = high.rolling(SWING_LOOKBACK).max().shift(1)
    swing_low  = low.rolling(SWING_LOOKBACK).min().shift(1)

    sweep_low_raw = (low < swing_low) & (close > swing_low) & (close > open_)
    sweep_high_raw = (high > swing_high) & (close < swing_high) & (close < open_)

    out["sweep_low"]  = sweep_low_raw.shift(1).fillna(False).astype(bool)
    out["sweep_high"] = sweep_high_raw.shift(1).fillna(False).astype(bool)

    return out


def _add_pin_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rejection / pin bar candles on 5M.
    Bullish pin bar: long lower wick, close > open.
    Bearish pin bar: long upper wick, close < open.
    Shifted by 1 for lag safety.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    open_ = out["Open"].astype(float)

    body       = (close - open_).abs()
    upper_wick = high - close.clip(lower=open_).combine(open_.clip(lower=close), max)
    lower_wick = close.clip(upper=open_).combine(open_.clip(upper=close), min) - low

    # Simpler wick calculation
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

    pin_bull = (lower_wick > 2 * body) & (lower_wick > upper_wick) & (close > open_)
    pin_bear = (upper_wick > 2 * body) & (upper_wick > lower_wick) & (close < open_)

    out["pin_bar_bull"] = pin_bull.shift(1).fillna(False).astype(bool)
    out["pin_bar_bear"] = pin_bear.shift(1).fillna(False).astype(bool)

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src_v2.data.loader import load_split_tf
    from pathlib import Path

    train_1h, _, _ = load_split_tf("1H")
    df_1h = add_1h_features(train_1h)
    print(f"1H features: {df_1h.shape[1]} columns, {len(df_1h)} rows")

    train_5m, _, _ = load_split_tf("5M")
    df_5m = add_5m_features(train_5m)
    print(f"5M features: {df_5m.shape[1]} columns, {len(df_5m)} rows")
