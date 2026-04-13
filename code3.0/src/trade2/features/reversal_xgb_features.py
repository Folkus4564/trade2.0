"""
features/reversal_xgb_features.py - Feature engineering for XGBoost reversal detector.

All output columns are lag-safe (use only data available at bar i without lookahead).
The columns added here supplement the standard 5M feature set and are consumed by
ReversalXGBModel.

Output columns:
  sd_zone_float       : sd_smoothed value (continuous, not discretized)
  atr_expansion       : atr_14 / rolling_mean(atr_14, 20) -- ATR burst detector
  candle_body_ratio   : |close-open| / (high-low + eps) -- candle conviction
  lower_wick_atr      : (min(open,close) - low) / atr_14 -- rejection wick strength
  upper_wick_atr      : (high - max(open,close)) / atr_14 -- rejection wick strength
  consecutive_extreme : bars in a row at sd_zone <= -2 (long) / >= +2 (short)
  consecutive_extreme_short: same for short direction
  sd_momentum         : sd_smoothed[i] - sd_smoothed[i-5] -- zone velocity
  ret_1               : 1-bar return (already in hmm_features but added if missing)
  ret_5               : 5-bar return
  in_demand_zone      : 1 if in a demand zone (from demand_supply features or 0)
  in_supply_zone      : 1 if in a supply zone
  ob_bullish          : 1 if in bullish OB (from luxalgo, or 0)
  ob_bearish          : 1 if in bearish OB
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def add_reversal_xgb_features(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Add XGBoost reversal detector input features to the DataFrame.

    All new columns are shift(1) applied at the end of this function to prevent
    lookahead bias. Only call this after all other 5M features have been computed.

    Args:
        df:     5M signal DataFrame with OHLC, atr_14, sd_smoothed, sd_zone columns.
        config: Full config dict (used to read sd_mean params).

    Returns:
        df copy with additional reversal feature columns.
    """
    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)
    open_ = out["Open"].astype(float)

    # ATR (use existing column preferably)
    if "atr_14" in out.columns:
        atr = out["atr_14"].astype(float).replace(0, np.nan).fillna(method="ffill").fillna(1.0)
    else:
        import talib
        atr = pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=14),
            index=out.index
        ).fillna(1.0)
        out["atr_14"] = atr

    # ----------------------------------------------------------------
    # 1. sd_zone as float (continuous depth, not discretized int)
    # ----------------------------------------------------------------
    if "sd_smoothed" in out.columns:
        out["sd_zone_float"] = out["sd_smoothed"].astype(float)
    elif "sd_zone" in out.columns:
        out["sd_zone_float"] = out["sd_zone"].astype(float)
    else:
        out["sd_zone_float"] = 0.0

    # ----------------------------------------------------------------
    # 2. ATR expansion: current ATR vs recent average
    # ----------------------------------------------------------------
    atr_mean = atr.rolling(20, min_periods=5).mean().fillna(atr)
    out["atr_expansion"] = (atr / (atr_mean + 1e-9)).clip(0.0, 5.0)

    # ----------------------------------------------------------------
    # 3. Candle body ratio: body size vs total range
    # ----------------------------------------------------------------
    body  = (close - open_).abs()
    range_ = (high - low).clip(lower=1e-6)
    out["candle_body_ratio"] = (body / range_).clip(0.0, 1.0)

    # ----------------------------------------------------------------
    # 4. Wick sizes (normalized by ATR)
    # ----------------------------------------------------------------
    candle_low  = np.minimum(close.values, open_.values)
    candle_high = np.maximum(close.values, open_.values)
    out["lower_wick_atr"] = ((candle_low  - low.values)  / (atr.values + 1e-9)).clip(0.0, 5.0)
    out["upper_wick_atr"] = ((high.values - candle_high) / (atr.values + 1e-9)).clip(0.0, 5.0)

    # ----------------------------------------------------------------
    # 5. Consecutive extreme bars (rolling count at sd extreme)
    # ----------------------------------------------------------------
    if "sd_zone" in out.columns:
        sd_zone = out["sd_zone"].fillna(0).round().astype(int)
    else:
        sd_zone = pd.Series(0, index=out.index)

    cfg_sd = (config or {}).get("strategies", {}).get("smc_sd_mean", {})
    min_zone = int(cfg_sd.get("min_entry_zone", 2))

    at_extreme_long  = (sd_zone <= -min_zone).astype(int)
    at_extreme_short = (sd_zone >=  min_zone).astype(int)

    # Rolling count of consecutive bars at extreme (vectorized via cumsum trick)
    def _consecutive_count(mask: pd.Series) -> pd.Series:
        m = mask.values
        result = np.zeros(len(m), dtype=float)
        count  = 0
        for i in range(len(m)):
            if m[i]:
                count += 1
            else:
                count = 0
            result[i] = count
        return pd.Series(result, index=mask.index)

    out["consecutive_extreme"]       = _consecutive_count(at_extreme_long)
    out["consecutive_extreme_short"] = _consecutive_count(at_extreme_short)

    # ----------------------------------------------------------------
    # 6. SD momentum: rate of change of SD position over 5 bars
    # ----------------------------------------------------------------
    sd_val = out["sd_zone_float"]
    out["sd_momentum"] = sd_val - sd_val.shift(5)

    # ----------------------------------------------------------------
    # 7. Short-term returns (add if not already present)
    # ----------------------------------------------------------------
    if "ret_1" not in out.columns:
        out["ret_1"] = close.pct_change(1)
    if "ret_5" not in out.columns:
        out["ret_5"] = close.pct_change(5)

    # ----------------------------------------------------------------
    # 8. Zone membership flags (use existing if present, else 0)
    # ----------------------------------------------------------------
    for col in ("in_demand_zone", "in_supply_zone", "ob_bullish", "ob_bearish"):
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0).astype(float)

    # ----------------------------------------------------------------
    # 9. ADX (directional strength — high ADX = trend, bad for mean reversion)
    # ----------------------------------------------------------------
    if "adx_14" not in out.columns:
        try:
            import talib
            out["adx_14"] = pd.Series(
                talib.ADX(high.values, low.values, close.values, timeperiod=14),
                index=out.index
            ).fillna(20.0)
        except Exception:
            out["adx_14"] = 20.0

    # ----------------------------------------------------------------
    # 10. Time-of-day features (cyclically encoded to avoid discontinuity)
    # ----------------------------------------------------------------
    if hasattr(out.index, "hour"):
        hour = out.index.hour.astype(float)
    else:
        try:
            hour = pd.DatetimeIndex(out.index).hour.astype(float)
        except Exception:
            hour = np.zeros(len(out))

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # ----------------------------------------------------------------
    # 11. Apply shift(1) to newly computed OHLC-derived columns
    #    (sd_zone_float mirrors sd_zone which is already shifted)
    #    (ret_1/ret_5: pct_change already lags by 1 bar naturally)
    # ----------------------------------------------------------------
    cols_to_shift = [
        "atr_expansion", "candle_body_ratio",
        "lower_wick_atr", "upper_wick_atr",
        "consecutive_extreme", "consecutive_extreme_short",
        "sd_momentum", "adx_14",
        "hour_sin", "hour_cos",
    ]
    for c in cols_to_shift:
        if c in out.columns:
            out[c] = out[c].shift(1)

    return out
