"""
features/hmm_features.py - HMM input feature extraction.
All HMM features are shift(1) to prevent lookahead.
"""

import numpy as np
import pandas as pd
import talib
from typing import Tuple

from trade2.features.indicators import hma


def add_hmm_features(
    df: pd.DataFrame,
    hma_period: int = 55,
    ema_period: int = 21,
    atr_period: int = 14,
    rsi_period: int = 14,
    adx_period: int = 14,
    dc_period:  int = 40,
) -> pd.DataFrame:
    """
    Add all 1H features needed for HMM regime detection.
    All hmm_feat_* columns are shift(1) to prevent lookahead.

    Returns df copy with added columns.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    vol   = out["Volume"].astype(float)

    # Returns
    out["log_ret"] = np.log(close / close.shift(1))
    out["ret_5"]   = close.pct_change(5)
    out["ret_20"]  = close.pct_change(20)

    # Trend
    out["hma"]       = hma(close, hma_period)
    out["hma_slope"] = out["hma"].diff(3) / (out["hma"].shift(3) + 1e-10)
    out["ema"]       = pd.Series(talib.EMA(close.values, timeperiod=ema_period), index=out.index)
    out["price_above_hma"] = (close > out["hma"]).astype(int)
    out["hma_rising"]      = (out["hma_slope"] > 0).astype(int)

    # Volatility
    out["atr_14"]   = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period), index=out.index
    )
    out["atr_norm"] = out["atr_14"] / (close + 1e-10)
    out["vol_20"]   = out["log_ret"].rolling(20).std() * np.sqrt(252 * 24)

    # Momentum
    out["rsi_14"]   = pd.Series(talib.RSI(close.values, timeperiod=rsi_period), index=out.index)
    out["rsi_norm"] = (out["rsi_14"] - 50.0) / 50.0

    # Trend strength
    out["adx_14"] = pd.Series(
        talib.ADX(high.values, low.values, close.values, timeperiod=adx_period), index=out.index
    )

    # MACD
    macd_v, sig_v, _ = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    out["macd"]      = pd.Series(macd_v, index=out.index)
    out["macd_sig"]  = pd.Series(sig_v,  index=out.index)
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close.values, timeperiod=20)
    out["bb_width"] = pd.Series((bb_upper - bb_lower) / (bb_mid + 1e-10), index=out.index)
    out["bb_pos"]   = pd.Series(
        (close.values - bb_lower) / (bb_upper - bb_lower + 1e-10), index=out.index
    )

    # Donchian Channel (shifted 1 bar)
    out["dc_upper"] = high.rolling(dc_period).max().shift(1)
    out["dc_lower"] = low.rolling(dc_period).min().shift(1)
    out["dc_mid"]   = (out["dc_upper"] + out["dc_lower"]) / 2.0
    out["breakout_long"]  = (close > out["dc_upper"]).astype(int)
    out["breakout_short"] = (close < out["dc_lower"]).astype(int)
    out["breakout_strength_long"]  = (close - out["dc_upper"]).clip(lower=0) / (out["atr_14"] + 1e-10)
    out["breakout_strength_short"] = (out["dc_lower"] - close).clip(lower=0) / (out["atr_14"] + 1e-10)

    atr_ma = out["atr_14"].rolling(20).mean()
    out["atr_expansion"] = (out["atr_14"] > atr_ma).astype(int)

    # HMM input features - ALL shifted 1 bar for lookahead safety
    out["hmm_feat_ret"]       = out["log_ret"].shift(1)
    out["hmm_feat_rsi"]       = out["rsi_norm"].shift(1)
    out["hmm_feat_atr"]       = out["atr_norm"].shift(1)
    out["hmm_feat_vol"]       = out["vol_20"].shift(1)
    out["hmm_feat_hma_slope"] = out["hma_slope"].shift(1)
    out["hmm_feat_bb_width"]  = out["bb_width"].shift(1)
    out["hmm_feat_macd"]      = out["macd_hist"].shift(1)

    return out


def get_hmm_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
    """Extract the HMM feature matrix (NaN-free rows) and corresponding index."""
    cols = [
        "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
        "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
    ]
    cols = [c for c in cols if c in df.columns]
    feat = df[cols].dropna()
    return feat.values, feat.index
