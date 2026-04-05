"""
features/sd_adaptive_mean.py - Adaptive SD Distance Mean indicator.

Python translation of the TradingView Pine Script:
  "Adaptive SD Distance Mean" (works best on M1, adapted here for M5/M15).

Logic:
  - ATR switches between a short and long SMA period (adaptive mean).
  - Standardized distance = (close - mean) / stdev * (atr / sma(atr)).
  - Smooth the standardized distance with EMA.
  - Dynamic SD bands scale with the ATR ratio.

All output columns are shift(1) for lag safety.

Output columns added to df:
  sd_mean         : adaptive SMA of close (short or long depending on ATR)
  sd_distance     : raw close - sd_mean
  sd_standardized : standardized and ATR-scaled distance
  sd_smoothed     : EMA-smoothed standardized distance (main oscillator)
  sd_band_1..4    : dynamic positive SD bands (symmetric: band_N and -band_N)
  sd_zone         : integer -4..+4 indicating which SD band the smoothed distance is in
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def compute_sd_adaptive_mean(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute Adaptive SD Distance Mean features and add them to df.

    Parameters read from config["strategies"]["smc_sd_mean"]:
      sd_length_short    : SMA period when ATR < threshold (default 14)
      sd_length_long     : SMA period when ATR >= threshold (default 50)
      sd_atr_length      : ATR period for the adaptive trigger (default 14)
      sd_atr_threshold   : ATR value that switches between short/long SMA (default 1.0)
      sd_smooth_length   : EMA period for smoothing the standardized distance (default 20)
      sd_smooth_sd_length: EMA period for smoothing the dynamic SD bands (default 50)

    All outputs are shift(1) to prevent lookahead bias.
    """
    cfg = (config or {}).get("strategies", {}).get("smc_sd_mean", {})

    length_short    = int(cfg.get("sd_length_short",     14))
    length_long     = int(cfg.get("sd_length_long",      50))
    atr_length      = int(cfg.get("sd_atr_length",       14))
    atr_threshold   = float(cfg.get("sd_atr_threshold",  1.0))
    smooth_length   = int(cfg.get("sd_smooth_length",    20))
    sd_smooth_len   = int(cfg.get("sd_smooth_sd_length", 50))

    out   = df.copy()
    close = out["Close"].astype(float)
    high  = out["High"].astype(float)
    low   = out["Low"].astype(float)

    # ATR (reuse existing column if available, otherwise compute)
    if "atr_14" in out.columns and atr_length == 14:
        atr = out["atr_14"].astype(float)
    else:
        atr = pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=atr_length),
            index=out.index,
        )

    # Adaptive SMA: switch between short/long based on ATR vs threshold
    # ATR < threshold -> use short period (low volatility, fast response)
    # ATR >= threshold -> use long period (high volatility, slow response)
    mean_short = close.rolling(length_short, min_periods=max(1, length_short // 2)).mean()
    mean_long  = close.rolling(length_long,  min_periods=max(1, length_long  // 2)).mean()
    use_short  = (atr < atr_threshold).values
    mean       = pd.Series(
        np.where(use_short, mean_short.values, mean_long.values),
        index=out.index,
    )

    # Adaptive stdev (same switch)
    std_short = close.rolling(length_short, min_periods=max(1, length_short // 2)).std()
    std_long  = close.rolling(length_long,  min_periods=max(1, length_long  // 2)).std()
    std       = pd.Series(
        np.where(use_short, std_short.values, std_long.values),
        index=out.index,
    )

    # Distance from mean and standardized version
    distance      = close - mean
    atr_sma       = atr.rolling(atr_length, min_periods=max(1, atr_length // 2)).mean()
    atr_ratio     = atr / (atr_sma + 1e-10)
    standardized  = (distance / (std + 1e-10)) * atr_ratio

    # EMA-smooth the standardized distance (main oscillator)
    std_filled = standardized.fillna(0.0).values
    smoothed   = pd.Series(
        talib.EMA(std_filled, timeperiod=smooth_length),
        index=out.index,
    )

    # Dynamic SD bands: EMA of (ATR ratio * N)
    ar_filled = atr_ratio.fillna(0.0).values
    band_1 = pd.Series(talib.EMA(ar_filled * 1.0, timeperiod=sd_smooth_len), index=out.index)
    band_2 = pd.Series(talib.EMA(ar_filled * 2.0, timeperiod=sd_smooth_len), index=out.index)
    band_3 = pd.Series(talib.EMA(ar_filled * 3.0, timeperiod=sd_smooth_len), index=out.index)
    band_4 = pd.Series(talib.EMA(ar_filled * 4.0, timeperiod=sd_smooth_len), index=out.index)

    # SD zone: which band does the smoothed distance sit in? (-4 to +4)
    sm = smoothed.values
    b1 = band_1.values
    b2 = band_2.values
    b3 = band_3.values
    b4 = band_4.values

    zone = np.zeros(len(sm), dtype=int)
    zone = np.where(sm >=  b4,  4, zone)
    zone = np.where((sm < b4) & (sm >=  b3),  3, zone)
    zone = np.where((sm < b3) & (sm >=  b2),  2, zone)
    zone = np.where((sm < b2) & (sm >=  b1),  1, zone)
    zone = np.where((sm < -b4),              -4, zone)
    zone = np.where((sm > -b4) & (sm <= -b3), -3, zone)
    zone = np.where((sm > -b3) & (sm <= -b2), -2, zone)
    zone = np.where((sm > -b2) & (sm <= -b1), -1, zone)

    # Write to df -- all shift(1) for lag safety
    out["sd_mean"]         = mean.shift(1)
    out["sd_distance"]     = distance.shift(1)
    out["sd_standardized"] = standardized.shift(1)
    out["sd_smoothed"]     = smoothed.shift(1)
    out["sd_band_1"]       = band_1.shift(1)
    out["sd_band_2"]       = band_2.shift(1)
    out["sd_band_3"]       = band_3.shift(1)
    out["sd_band_4"]       = band_4.shift(1)
    out["sd_zone"]         = pd.Series(zone, index=out.index).shift(1).fillna(0).astype(int)

    return out
