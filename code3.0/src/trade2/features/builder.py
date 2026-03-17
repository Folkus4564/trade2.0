"""
features/builder.py - Orchestrates feature construction from config.
Calls indicators + SMC, returns fully featured DataFrames.
"""

import pandas as pd
import talib
from typing import Dict, Any

from trade2.features.hmm_features import add_hmm_features
from trade2.features.smc import add_smc_features, add_pin_bar_features


def add_1h_features(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Build all 1H features (HMM inputs + SMC on 1H bars).

    Args:
        df:     1H OHLCV DataFrame
        config: Config dict (from load_config). If None, uses defaults.

    Returns:
        DataFrame with all feature columns.
    """
    feat_cfg = (config or {}).get("features", {})
    smc_cfg  = (config or {}).get("smc", {})

    df_feat = add_hmm_features(
        df,
        hma_period=feat_cfg.get("hma_period", 55),
        ema_period=feat_cfg.get("ema_period", 21),
        atr_period=feat_cfg.get("atr_period", 14),
        rsi_period=feat_cfg.get("rsi_period", 14),
        adx_period=feat_cfg.get("adx_period", 14),
        dc_period =feat_cfg.get("dc_period",  40),
    )

    df_feat = add_smc_features(
        df_feat,
        ob_validity     = smc_cfg.get("ob_validity_bars",    20),
        fvg_validity    = smc_cfg.get("fvg_validity_bars",   15),
        swing_lookback  = smc_cfg.get("swing_lookback_bars", 20),
        ob_impulse_bars = smc_cfg.get("ob_impulse_bars",     3),
        ob_impulse_mult = smc_cfg.get("ob_impulse_mult",     1.5),
        atr_period      = feat_cfg.get("atr_period",         14),
    )

    if (config or {}).get("smc_luxalgo", {}).get("enabled", False):
        from trade2.features.smc_luxalgo import add_luxalgo_smc_features
        df_feat = add_luxalgo_smc_features(df_feat, config, config_key="smc_luxalgo")

    if (config or {}).get("strategies", {}).get("cdc", {}).get("enabled", False):
        from trade2.features.cdc import add_cdc_features
        df_feat = add_cdc_features(df_feat, config)

    if (config or {}).get("tv_indicators"):
        from trade2.features.tv_indicators import apply_tv_indicators
        df_feat = apply_tv_indicators(df_feat, config)

    return df_feat


def add_5m_features(df: pd.DataFrame, config: Dict[str, Any] = None, dc_period: int = None) -> pd.DataFrame:
    """
    Build all 5M features (SMC entry signals + supporting indicators).

    Args:
        df:        5M OHLCV DataFrame
        config:    Config dict. Uses smc_5m section for SMC params.
        dc_period: Override for Donchian period (for optimization).

    Returns:
        DataFrame with SMC, ATR, ADX, RSI, Donchian features.
    """
    import numpy as np

    feat_cfg = (config or {}).get("features", {})
    smc_cfg  = (config or {}).get("smc_5m", {})

    atr_period = feat_cfg.get("atr_period", 14)
    adx_period = feat_cfg.get("adx_period", 14)
    rsi_period = feat_cfg.get("rsi_period", 14)
    dc_p       = dc_period if dc_period is not None else feat_cfg.get("dc_period", 40)

    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)

    # Basic indicators on 5M
    out["atr_14"]   = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period), index=out.index
    )
    out["atr_norm"] = out["atr_14"] / (close + 1e-10)
    out["adx_14"]   = pd.Series(
        talib.ADX(high.values, low.values, close.values, timeperiod=adx_period), index=out.index
    )
    out["rsi_14"]   = pd.Series(
        talib.RSI(close.values, timeperiod=rsi_period), index=out.index
    )

    # Donchian channel (shifted 1 bar)
    out["dc_upper"] = high.rolling(dc_p).max().shift(1)
    out["dc_lower"] = low.rolling(dc_p).min().shift(1)
    out["dc_mid"]   = (out["dc_upper"] + out["dc_lower"]) / 2.0
    out["breakout_long"]  = (close > out["dc_upper"]).astype(int)
    out["breakout_short"] = (close < out["dc_lower"]).astype(int)

    atr_ma = out["atr_14"].rolling(20).mean()
    out["atr_expansion"] = (out["atr_14"] > atr_ma).astype(int)

    # Bollinger Band position (for range strategy: 0=at lower BB, 1=at upper BB)
    import numpy as np
    bb_period = feat_cfg.get("bb_period_5m", 60)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close.values, timeperiod=bb_period)
    out["bb_pos"] = (close.values - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # SMC features (with 5M-specific validity periods)
    out = add_smc_features(
        out,
        ob_validity     = smc_cfg.get("ob_validity_bars",    60),
        fvg_validity    = smc_cfg.get("fvg_validity_bars",   36),
        swing_lookback  = smc_cfg.get("swing_lookback_bars", 60),
        ob_impulse_bars = smc_cfg.get("ob_impulse_bars",     3),
        ob_impulse_mult = smc_cfg.get("ob_impulse_mult",     1.5),
        atr_period      = atr_period,
    )

    # Pin bar features
    out = add_pin_bar_features(out)

    if (config or {}).get("smc_luxalgo_5m", {}).get("enabled", False):
        from trade2.features.smc_luxalgo import add_luxalgo_smc_features
        out = add_luxalgo_smc_features(out, config, config_key="smc_luxalgo_5m")

    if (config or {}).get("strategies", {}).get("cdc", {}).get("enabled", False):
        from trade2.features.cdc import add_cdc_features
        out = add_cdc_features(out, config)

    if (config or {}).get("tv_indicators"):
        from trade2.features.tv_indicators import apply_tv_indicators
        out = apply_tv_indicators(out, config)

    return out
