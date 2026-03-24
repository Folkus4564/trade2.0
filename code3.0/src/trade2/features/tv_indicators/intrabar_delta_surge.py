"""
features/intrabar_delta_surge.py - Intrabar Delta Surge feature detection.
Computes intrabar pressure (close position within bar range) momentum.
Bull when pressure sharply shifts toward close near highs; bear on opposite.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_intrabar_delta_surge_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Intrabar Delta Surge features.

    Core concept:
      - Intrabar delta = (Close - Open) / (High - Low + epsilon)
        Measures where close landed relative to the bar's full range, signed by open-to-close direction.
      - Pressure = (Close - Low) / (High - Low + epsilon)
        Where in the bar did price close (0=low, 1=high).
      - Delta Surge = EMA(pressure, period) - EMA(pressure, period*2)
        Fast vs slow pressure momentum.
      - Surge magnitude smoothed and thresholded.

    Zones:
      bull: surge > threshold  (close clustering near highs, accelerating)
      bear: surge < -threshold (close clustering near lows, accelerating)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["intrabar_delta_surge"].

    Returns:
        df copy with intrabar_delta_surge_ columns added.
    """
    cfg = config.get("tv_indicators", {}).get("intrabar_delta_surge", {})
    period    = int(cfg.get("period", 3))
    threshold = float(cfg.get("threshold", 0.3))

    # Clamp periods for scalping safety
    period = max(3, min(period, 14))
    slow_period = min(period * 2, 14)

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values
    open_ = out["Open"].astype(float).values

    eps = 1e-10

    # --- Bar range ---
    bar_range = high - low + eps

    # --- Intrabar pressure: where did close land in [low, high] ---
    # 0 = closed at low, 1 = closed at high
    pressure = (close - low) / bar_range  # [0, 1]

    # --- Intrabar delta: directional body relative to full range ---
    # Positive = bullish body, negative = bearish body
    delta = (close - open_) / bar_range   # [-1, 1]

    # --- Combined signal: pressure weighted by delta direction ---
    # When delta is positive and pressure is high -> strong bull intrabar surge
    # When delta is negative and pressure is low  -> strong bear intrabar surge
    combined = pressure * np.sign(delta + eps)  # blend pressure with direction

    # --- Fast and slow EMA of pressure (talib) ---
    fast_ema = talib.EMA(pressure, timeperiod=period)
    slow_ema = talib.EMA(pressure, timeperiod=slow_period)

    # --- Surge: fast minus slow (momentum of pressure) ---
    surge = fast_ema - slow_ema  # positive = pressure accelerating toward highs

    # --- Smooth delta for noise reduction ---
    smoothed_delta = talib.EMA(delta, timeperiod=period)

    # --- Rate of change of surge (surge acceleration) ---
    # Use talib ROCP (rate of change percentage) over short window
    surge_roc = talib.ROCP(np.where(np.isnan(surge), 0.0, surge).astype(float), timeperiod=period)

    # --- Final signal: surge above/below threshold AND delta agrees ---
    bull_raw = (surge > threshold) & (smoothed_delta > 0.0)
    bear_raw = (surge < -threshold) & (smoothed_delta < 0.0)

    # --- Additional confirmation: surge accelerating in same direction ---
    bull_raw = bull_raw & (surge_roc > 0.0)
    bear_raw = bear_raw & (surge_roc < 0.0)

    # Replace NaN-induced Falses cleanly
    valid_mask = (
        ~np.isnan(fast_ema) &
        ~np.isnan(slow_ema) &
        ~np.isnan(smoothed_delta) &
        ~np.isnan(surge_roc)
    )
    bull_raw = bull_raw & valid_mask
    bear_raw = bear_raw & valid_mask

    # --- Helper: numpy 1-bar lag ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty(len(arr), dtype=bool)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    idx = out.index

    # --- All outputs shifted(1) for lag safety ---
    out["intrabar_delta_surge_pressure"]       = pd.Series(_lag1(pressure),        index=idx, dtype=float)
    out["intrabar_delta_surge_delta"]          = pd.Series(_lag1(delta),           index=idx, dtype=float)
    out["intrabar_delta_surge_surge"]          = pd.Series(
        pd.Series(surge, index=idx).shift(1)
    )
    out["intrabar_delta_surge_smoothed_delta"] = pd.Series(
        pd.Series(smoothed_delta, index=idx).shift(1)
    )
    out["intrabar_delta_surge_bull"]           = pd.Series(_lag1(bull_raw), index=idx, dtype=bool)
    out["intrabar_delta_surge_bear"]           = pd.Series(_lag1(bear_raw), index=idx, dtype=bool)

    return out