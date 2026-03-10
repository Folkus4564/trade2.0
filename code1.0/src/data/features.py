"""
Module: features.py
Purpose: Feature engineering for XAUUSD systematic strategy - all features lag-safe
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import numpy as np
import pandas as pd
import talib
from typing import Tuple

from src.config import get_config


# ── Smart Money Concepts (SMC) Features ───────────────────────────────────────

def compute_atr_pandas(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute ATR using pandas only (no TA-Lib dependency for SMC functions).
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_smc_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Add Smart Money Concepts (SMC) features to the DataFrame.

    All features are lag-safe: signals are based on completed bars only.
    The resulting boolean columns are shifted by 1 so they represent
    whether the PREVIOUS bar triggered the condition.

    Features added:
      ob_bullish  - price is retesting a valid bullish Order Block
      ob_bearish  - price is retesting a valid bearish Order Block
      fvg_bullish - price is inside a valid bullish Fair Value Gap
      fvg_bearish - price is inside a valid bearish Fair Value Gap
      sweep_low   - bullish liquidity sweep (wick below swing low, close above)
      sweep_high  - bearish liquidity sweep (wick above swing high, close below)

    Args:
        df:         OHLCV DataFrame (Open, High, Low, Close, Volume)
        atr_period: ATR lookback for impulse detection

    Returns:
        df copy with SMC feature columns added
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

    # ── Order Blocks ──────────────────────────────────────────────────────────
    # Impulse threshold: >= 1.5 * ATR(14) move within 3 consecutive bars
    # Bullish OB: last red candle before a bullish impulse of >= 1.5 * ATR
    # Bearish OB: last green candle before a bearish impulse of >= 1.5 * ATR
    # Valid for 20 bars. Retest: current close enters OB zone.
    # All references use already-closed bars to avoid lookahead.

    cfg = get_config()
    smc = cfg.get("smc", {})
    OB_VALIDITY  = smc.get("ob_validity_bars",    20)
    IMPULSE_BARS = smc.get("ob_impulse_bars",      3)
    IMPULSE_MULT = smc.get("ob_impulse_mult",    1.5)

    # Store OB zone boundaries for each bar (computed on prior bars only)
    bull_ob_low  = np.full(n, np.nan)
    bull_ob_high = np.full(n, np.nan)
    bear_ob_low  = np.full(n, np.nan)
    bear_ob_high = np.full(n, np.nan)

    # We scan completed bars to detect OBs; lag is built in by scanning i < current_bar
    for i in range(IMPULSE_BARS + 1, n):
        # Check if there is a bullish impulse ending at bar i-1
        # impulse = close[i-1] - low[i-IMPULSE_BARS]  (simplified: net move over window)
        impulse_up   = close_arr[i - 1] - min(low_arr[i - IMPULSE_BARS: i])
        impulse_down = max(high_arr[i - IMPULSE_BARS: i]) - close_arr[i - 1]
        threshold    = IMPULSE_MULT * atr_arr[i - 1] if not np.isnan(atr_arr[i - 1]) else 0.0

        if threshold <= 0:
            continue

        if impulse_up >= threshold:
            # Find the last red (bearish) candle before bar i-1 within the impulse window
            for j in range(i - 1, max(i - IMPULSE_BARS - 1, -1), -1):
                if open_arr[j] > close_arr[j]:   # bearish candle
                    # This is the bullish OB; it is valid starting from bar j+1
                    # For the current bar i, mark the active OB zone if within OB_VALIDITY
                    if (i - j) <= OB_VALIDITY:
                        bull_ob_low[i]  = low_arr[j]
                        bull_ob_high[i] = high_arr[j]
                    break

        if impulse_down >= threshold:
            # Find the last green (bullish) candle before bar i-1 within the impulse window
            for j in range(i - 1, max(i - IMPULSE_BARS - 1, -1), -1):
                if close_arr[j] > open_arr[j]:   # bullish candle
                    if (i - j) <= OB_VALIDITY:
                        bear_ob_low[i]  = low_arr[j]
                        bear_ob_high[i] = high_arr[j]
                    break

    # Retest condition: current close enters OB zone
    bull_ob_retest_raw = (
        (~np.isnan(bull_ob_low)) &
        (close_arr >= bull_ob_low) &
        (close_arr <= bull_ob_high)
    )
    bear_ob_retest_raw = (
        (~np.isnan(bear_ob_low)) &
        (close_arr >= bear_ob_low) &
        (close_arr <= bear_ob_high)
    )

    # Shift by 1: signal at bar i means it was detected at bar i-1
    out["ob_bullish"] = pd.Series(bull_ob_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)
    out["ob_bearish"] = pd.Series(bear_ob_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)

    # ── Fair Value Gaps (FVG) ─────────────────────────────────────────────────
    # Bullish FVG: bar[i-2].high < bar[i].low  -> gap zone: [bar[i-2].high, bar[i].low]
    # Bearish FVG: bar[i-2].low  > bar[i].high -> gap zone: [bar[i].high, bar[i-2].low]
    # Valid for 15 bars. Entry: close inside gap zone.
    # Lookahead prevention: detect FVG on bars i-2..i (all completed), carry forward zone.

    FVG_VALIDITY = smc.get("fvg_validity_bars", 15)

    bull_fvg_lo  = np.full(n, np.nan)
    bull_fvg_hi  = np.full(n, np.nan)
    bear_fvg_lo  = np.full(n, np.nan)
    bear_fvg_hi  = np.full(n, np.nan)

    # Last known active FVG boundaries (carry forward until invalidated)
    active_bull_fvg_lo  = np.nan
    active_bull_fvg_hi  = np.nan
    active_bull_fvg_age = 0

    active_bear_fvg_lo  = np.nan
    active_bear_fvg_hi  = np.nan
    active_bear_fvg_age = 0

    for i in range(2, n):
        # Detect new FVG at bar i (using bar i-2 and bar i, both completed at bar i)
        # To avoid lookahead on bar i, we detect at bar i-1 relative to signal bar
        # -> shift happens below; here we detect using bar i-2 and i (already closed)
        h_im2 = high_arr[i - 2]
        l_im2 = low_arr[i - 2]
        l_i   = low_arr[i]
        h_i   = high_arr[i]

        # Bullish FVG
        if h_im2 < l_i:   # gap exists
            # New bullish FVG detected at bar i
            active_bull_fvg_lo  = h_im2
            active_bull_fvg_hi  = l_i
            active_bull_fvg_age = 0

        # Bearish FVG
        if l_im2 > h_i:   # gap exists
            active_bear_fvg_lo  = h_i
            active_bear_fvg_hi  = l_im2
            active_bear_fvg_age = 0

        # Carry-forward active FVGs and check validity / invalidation
        if not np.isnan(active_bull_fvg_lo):
            active_bull_fvg_age += 1
            # Invalidated if price closes below FVG lower boundary
            if close_arr[i] < active_bull_fvg_lo or active_bull_fvg_age > FVG_VALIDITY:
                active_bull_fvg_lo  = np.nan
                active_bull_fvg_hi  = np.nan
                active_bull_fvg_age = 0
            else:
                bull_fvg_lo[i] = active_bull_fvg_lo
                bull_fvg_hi[i] = active_bull_fvg_hi

        if not np.isnan(active_bear_fvg_lo):
            active_bear_fvg_age += 1
            # Invalidated if price closes above FVG upper boundary
            if close_arr[i] > active_bear_fvg_hi or active_bear_fvg_age > FVG_VALIDITY:
                active_bear_fvg_lo  = np.nan
                active_bear_fvg_hi  = np.nan
                active_bear_fvg_age = 0
            else:
                bear_fvg_lo[i] = active_bear_fvg_lo
                bear_fvg_hi[i] = active_bear_fvg_hi

    # Retest: current close is inside the gap zone
    bull_fvg_retest_raw = (
        (~np.isnan(bull_fvg_lo)) &
        (close_arr >= bull_fvg_lo) &
        (close_arr <= bull_fvg_hi)
    )
    bear_fvg_retest_raw = (
        (~np.isnan(bear_fvg_lo)) &
        (close_arr >= bear_fvg_lo) &
        (close_arr <= bear_fvg_hi)
    )

    # Shift by 1 for lag safety
    out["fvg_bullish"] = pd.Series(bull_fvg_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)
    out["fvg_bearish"] = pd.Series(bear_fvg_retest_raw, index=out.index).shift(1).fillna(False).astype(bool)

    # ── Liquidity Sweeps ──────────────────────────────────────────────────────
    # sweep_low:  wick below 20-bar rolling min of lows (excl. current bar), close above it
    # sweep_high: wick above 20-bar rolling max of highs (excl. current bar), close below it
    # Reversal confirmation: close direction
    # Shift by 1 for lag safety.

    SWING_LOOKBACK = smc.get("swing_lookback_bars", 20)

    # Rolling max/min excluding current bar (shift(1) on rolling result)
    swing_high = high.rolling(SWING_LOOKBACK).max().shift(1)
    swing_low  = low.rolling(SWING_LOOKBACK).min().shift(1)

    # Bullish sweep: low wicks below swing_low, close above it, bullish close
    sweep_low_raw = (
        (low  < swing_low) &
        (close > swing_low) &
        (close > open_)
    )

    # Bearish sweep: high wicks above swing_high, close below it, bearish close
    sweep_high_raw = (
        (high  > swing_high) &
        (close < swing_high) &
        (close < open_)
    )

    # Shift by 1 for lag safety
    out["sweep_low"]  = sweep_low_raw.shift(1).fillna(False).astype(bool)
    out["sweep_high"] = sweep_high_raw.shift(1).fillna(False).astype(bool)

    return out


# ── Hull Moving Average ────────────────────────────────────────────────────────

def hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).
    Reduces lag while remaining smooth.
    """
    half        = max(int(period / 2), 1)
    sqrt_p      = max(int(np.sqrt(period)), 1)
    wma_half    = talib.WMA(series.values.astype(float), timeperiod=half)
    wma_full    = talib.WMA(series.values.astype(float), timeperiod=period)
    raw         = 2 * wma_half - wma_full
    result      = talib.WMA(raw, timeperiod=sqrt_p)
    return pd.Series(result, index=series.index, name=f"HMA_{period}")


# ── Core Feature Engineering ───────────────────────────────────────────────────

def add_features(
    df: pd.DataFrame,
    hma_period: int = None,
    ema_period: int = None,
    atr_period: int = None,
    rsi_period: int = None,
    adx_period: int = None,
    dc_period: int = None,
) -> pd.DataFrame:
    """
    Add all strategy features to an OHLCV DataFrame.

    LOOKAHEAD SAFETY: All HMM-input features are .shift(1) so the model
    never sees the current bar's data when predicting the current bar's regime.
    Donchian channel levels are also shifted(1) so entry signals are based on
    the previous bar's completed range.

    Args:
        df: OHLCV DataFrame with columns Open, High, Low, Close, Volume
        hma_period: Hull MA period
        ema_period: EMA period
        atr_period: ATR lookback
        rsi_period: RSI lookback
        adx_period: ADX lookback
        dc_period:  Donchian Channel lookback (breakout window)

    Returns:
        df copy with all features added
    """
    cfg = get_config()
    feat_cfg = cfg.get("features", {})
    if hma_period is None: hma_period = feat_cfg.get("hma_period", 55)
    if ema_period is None: ema_period = feat_cfg.get("ema_period", 21)
    if atr_period is None: atr_period = feat_cfg.get("atr_period", 14)
    if rsi_period is None: rsi_period = feat_cfg.get("rsi_period", 14)
    if adx_period is None: adx_period = feat_cfg.get("adx_period", 14)
    if dc_period  is None: dc_period  = feat_cfg.get("dc_period",  40)

    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    vol   = out["Volume"].astype(float)

    # ── Returns ───────────────────────────────────────────────────────────────
    out["log_ret"]   = np.log(close / close.shift(1))
    out["ret_5"]     = close.pct_change(5)
    out["ret_20"]    = close.pct_change(20)

    # ── Trend indicators ──────────────────────────────────────────────────────
    out["hma"]       = hma(close, hma_period)
    out["hma_slope"] = out["hma"].diff(3) / (out["hma"].shift(3) + 1e-10)
    out["ema"]       = pd.Series(
        talib.EMA(close.values, timeperiod=ema_period), index=out.index
    )
    out["price_above_hma"] = (close > out["hma"]).astype(int)
    out["hma_rising"]      = (out["hma_slope"] > 0).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    out["atr_14"]    = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period),
        index=out.index
    )
    out["atr_norm"]  = out["atr_14"] / (close + 1e-10)
    out["vol_20"]    = out["log_ret"].rolling(20).std() * np.sqrt(252 * 24)

    # ── Momentum ──────────────────────────────────────────────────────────────
    out["rsi_14"]    = pd.Series(
        talib.RSI(close.values, timeperiod=rsi_period), index=out.index
    )
    out["rsi_norm"]  = (out["rsi_14"] - 50.0) / 50.0   # Range: -1 to +1

    # ── Trend strength ────────────────────────────────────────────────────────
    out["adx_14"]    = pd.Series(
        talib.ADX(high.values, low.values, close.values, timeperiod=adx_period),
        index=out.index
    )

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_v, sig_v, _ = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    out["macd"]      = pd.Series(macd_v, index=out.index)
    out["macd_sig"]  = pd.Series(sig_v,  index=out.index)
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close.values, timeperiod=20)
    out["bb_width"]  = pd.Series((bb_upper - bb_lower) / (bb_mid + 1e-10), index=out.index)
    out["bb_pos"]    = pd.Series(
        (close.values - bb_lower) / (bb_upper - bb_lower + 1e-10), index=out.index
    )

    # ── Donchian Channel Breakout ─────────────────────────────────────────────
    # Shift 1 bar: entry signal is based on previous bar's completed N-bar range
    out["dc_upper"] = high.rolling(dc_period).max().shift(1)
    out["dc_lower"] = low.rolling(dc_period).min().shift(1)
    out["dc_mid"]   = (out["dc_upper"] + out["dc_lower"]) / 2.0

    # Breakout flags: current close breaks above/below previous N-bar channel
    out["breakout_long"]  = (close > out["dc_upper"]).astype(int)
    out["breakout_short"] = (close < out["dc_lower"]).astype(int)

    # Breakout strength: how far price extended beyond channel (normalized by ATR)
    out["breakout_strength_long"]  = (close - out["dc_upper"]).clip(lower=0) / (out["atr_14"] + 1e-10)
    out["breakout_strength_short"] = (out["dc_lower"] - close).clip(lower=0) / (out["atr_14"] + 1e-10)

    # ATR expansion: current ATR above 20-bar ATR average (confirms volatility surge on breakout)
    atr_ma = out["atr_14"].rolling(20).mean()
    out["atr_expansion"] = (out["atr_14"] > atr_ma).astype(int)

    # ── HMM input features (shifted 1 bar to prevent lookahead) ──────────────
    out["hmm_feat_ret"]       = out["log_ret"].shift(1)
    out["hmm_feat_rsi"]       = out["rsi_norm"].shift(1)
    out["hmm_feat_atr"]       = out["atr_norm"].shift(1)
    out["hmm_feat_vol"]       = out["vol_20"].shift(1)
    out["hmm_feat_hma_slope"] = out["hma_slope"].shift(1)
    out["hmm_feat_bb_width"]  = out["bb_width"].shift(1)
    out["hmm_feat_macd"]      = out["macd_hist"].shift(1)

    # ── SMC features ──────────────────────────────────────────────────────────
    out = add_smc_features(out, atr_period=atr_period)

    return out


def get_hmm_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
    """
    Extract the HMM feature matrix (rows without NaN) and corresponding index.
    7 features: log_ret, rsi, atr, vol, hma_slope, bb_width, macd_hist
    """
    cols = [
        "hmm_feat_ret",
        "hmm_feat_rsi",
        "hmm_feat_atr",
        "hmm_feat_vol",
        "hmm_feat_hma_slope",
        "hmm_feat_bb_width",
        "hmm_feat_macd",
    ]
    # Only use cols that exist (backward compat)
    cols = [c for c in cols if c in df.columns]
    feat = df[cols].dropna()
    return feat.values, feat.index


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.loader import load_split

    train, val, test = load_split("1H")
    df = add_features(train)
    print("Feature columns:", [c for c in df.columns if c not in ["Open","High","Low","Close","Volume"]])
    print(df[["log_ret", "hma", "rsi_14", "atr_14", "adx_14"]].tail(5))
    print(f"Shape: {df.shape}")
