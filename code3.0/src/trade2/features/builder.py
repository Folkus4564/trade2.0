"""
features/builder.py - Orchestrates feature construction from config.
Calls indicators + SMC, returns fully featured DataFrames.
"""

import pandas as pd
import talib
from typing import Dict, Any

from trade2.features.hmm_features import add_hmm_features
from trade2.features.smc import add_smc_features, add_pin_bar_features


def add_5m_full_features(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Combined 5M feature set for single-TF mode (regime_timeframe = 5M).

    Calls add_5m_features() to build all signal-TF features (SMC, SD mean, etc.),
    then calls add_hmm_features() on the result to add the HMM input columns (hmm_feat_*).

    This is used by run_pipeline.py when strategy.mode = 'single_tf' and
    regime_timeframe is a sub-hourly timeframe (5M, 15M).
    """
    from trade2.features.hmm_features import add_hmm_features

    feat_cfg = (config or {}).get("features", {})
    # Step 1: 5M signal features (SMC, LuxAlgo, SD adaptive mean, demand/supply, reversal patterns)
    out = add_5m_features(df, config)
    # Step 2: HMM input columns (hmm_feat_ret, hmm_feat_rsi, etc.)
    # This also picks up sd_smoothed and pd_ratio columns added in step 1
    # to populate hmm_feat_sd_distance and hmm_feat_pd_ratio.
    out = add_hmm_features(
        out,
        hma_period=feat_cfg.get("hma_period", 55),
        ema_period=feat_cfg.get("ema_period", 21),
        atr_period=feat_cfg.get("atr_period", 14),
        rsi_period=feat_cfg.get("rsi_period", 14),
        adx_period=feat_cfg.get("adx_period", 14),
        dc_period =feat_cfg.get("dc_period",  40),
    )
    return out


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

    if smc_cfg.get("enabled", True):
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

    _cdc_needed = (
        (config or {}).get("strategies", {}).get("cdc", {}).get("enabled", False) or
        (config or {}).get("strategies", {}).get("cdc_retest", {}).get("enabled", False)
    )
    if _cdc_needed:
        from trade2.features.cdc import add_cdc_features
        df_feat = add_cdc_features(df_feat, config)

    if (config or {}).get("tv_indicators"):
        from trade2.features.tv_indicators import apply_tv_indicators
        df_feat = apply_tv_indicators(df_feat, config)

    if (config or {}).get("strategies", {}).get("scalp_momentum", {}).get("enabled", False):
        from trade2.features.scalp_momentum import add_scalp_momentum_features
        df_feat = add_scalp_momentum_features(df_feat, config)

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

    atr_exp_lookback = feat_cfg.get("atr_expansion_lookback", 20)
    atr_exp_ratio    = feat_cfg.get("atr_expansion_ratio",    1.0)
    atr_ma = out["atr_14"].rolling(atr_exp_lookback, min_periods=1).mean()
    out["atr_expansion"] = (out["atr_14"] > atr_ma * atr_exp_ratio).astype(int)

    # Bollinger Band position (for range strategy: 0=at lower BB, 1=at upper BB)
    import numpy as np
    bb_period = feat_cfg.get("bb_period_5m", 60)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close.values, timeperiod=bb_period)
    out["bb_pos"] = (close.values - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # SMC features (with 5M-specific validity periods)
    if smc_cfg.get("enabled", True):
        out = add_smc_features(
            out,
            ob_validity     = smc_cfg.get("ob_validity_bars",    60),
            fvg_validity    = smc_cfg.get("fvg_validity_bars",   36),
            swing_lookback  = smc_cfg.get("swing_lookback_bars", 60),
            ob_impulse_bars = smc_cfg.get("ob_impulse_bars",     3),
            ob_impulse_mult = smc_cfg.get("ob_impulse_mult",     1.5),
            atr_period      = atr_period,
        )
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

    if (config or {}).get("strategies", {}).get("scalp_momentum", {}).get("enabled", False):
        from trade2.features.scalp_momentum import add_scalp_momentum_features
        out = add_scalp_momentum_features(out, config)

    if (config or {}).get("strategies", {}).get("smc_pullback_reversal", {}).get("enabled", False):
        from trade2.features.demand_supply import add_demand_supply_features
        from trade2.features.reversal_patterns import add_reversal_pattern_features
        ds_cfg = config["demand_supply_5m"]
        out = add_demand_supply_features(
            out,
            base_min_bars = ds_cfg["base_min_bars"],
            base_max_bars = ds_cfg["base_max_bars"],
            impulse_mult  = ds_cfg["impulse_mult"],
            validity_bars = ds_cfg["validity_bars"],
            atr_period    = feat_cfg.get("atr_period", 14),
        )
        out = add_reversal_pattern_features(out)

    # smc_ob_reversal: add reversal patterns (pin bar, engulfing) on 5M
    # Note: OB features (ob_bullish, struct_ob_bull_retest) come from smc_luxalgo_5m
    # which is already added above when smc_luxalgo_5m.enabled is true.
    if (config or {}).get("strategies", {}).get("smc_ob_reversal", {}).get("enabled", False):
        from trade2.features.reversal_patterns import add_reversal_pattern_features
        out = add_reversal_pattern_features(out)

    # smc_sd_mean: SD Adaptive Mean indicator + reversal patterns + demand/supply zones
    if (config or {}).get("strategies", {}).get("smc_sd_mean", {}).get("enabled", False):
        from trade2.features.sd_adaptive_mean import compute_sd_adaptive_mean
        out = compute_sd_adaptive_mean(out, config)
        # Reversal patterns (pin bar, engulfing) for entry confirmation
        from trade2.features.reversal_patterns import add_reversal_pattern_features
        out = add_reversal_pattern_features(out)
        # Demand/Supply zones (if not already added by smc_pullback_reversal)
        if "dz_demand_retest" not in out.columns:
            from trade2.features.demand_supply import add_demand_supply_features
            ds_cfg   = (config or {}).get("demand_supply_5m", {})
            feat_cfg = (config or {}).get("features", {})
            out = add_demand_supply_features(
                out,
                base_min_bars = ds_cfg.get("base_min_bars", 2),
                base_max_bars = ds_cfg.get("base_max_bars", 5),
                impulse_mult  = ds_cfg.get("impulse_mult",  1.5),
                validity_bars = ds_cfg.get("validity_bars",  60),
                atr_period    = feat_cfg.get("atr_period",   14),
            )
        # XGBoost reversal detector features (added last so all upstream cols present)
        xgb_cfg = (config or {}).get("strategies", {}).get("smc_sd_mean", {}).get("reversal_xgb", {})
        if xgb_cfg.get("enabled", False):
            from trade2.features.reversal_xgb_features import add_reversal_xgb_features
            out = add_reversal_xgb_features(out, config)

    return out
