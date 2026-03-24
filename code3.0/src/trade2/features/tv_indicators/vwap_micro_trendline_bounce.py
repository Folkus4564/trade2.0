"""
features/vwap_micro_trendline_bounce.py - VWAP Micro Trendline Bounce feature detection.
Detects price pullbacks to VWAP and a micro trendline, then breaks back in trend direction.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_vwap_micro_trendline_bounce_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute VWAP Micro Trendline Bounce features.

    Logic:
      1. Compute intraday VWAP (rolling approximation using volume-weighted price).
      2. Compute a micro trendline using a short linear regression of closes.
      3. Detect pullback to VWAP + trendline zone, then breakout in trend direction.

    Zones:
      bull: price pulled back near VWAP/trendline from above and now breaks up
      bear: price pulled back near VWAP/trendline from below and now breaks down

    Args:
        df:     OHLCV DataFrame
        config: Full config dict. Reads config["tv_indicators"]["vwap_micro_trendline_bounce"].

    Returns:
        df copy with vwap_micro_trendline_bounce_ columns added.
    """
    cfg = config.get("tv_indicators", {}).get("vwap_micro_trendline_bounce", {})
    confirm_bars = int(cfg.get("confirm_bars", 1))
    lookback     = int(cfg.get("lookback", 6))

    # Clamp to scalping constraints
    lookback     = max(3, min(lookback, 12))
    confirm_bars = max(1, min(confirm_bars, 5))

    out    = df.copy()
    close  = out["Close"].astype(float).values
    high   = out["High"].astype(float).values
    low    = out["Low"].astype(float).values
    open_  = out["Open"].astype(float).values
    volume = out["Volume"].astype(float).values
    n      = len(close)

    # ── 1. Rolling VWAP (anchored to rolling window, not session) ──────────────
    # typical_price * volume / rolling_volume_sum
    typical = (high + low + close) / 3.0
    tp_vol  = typical * volume

    roll_tp_vol = np.full(n, np.nan)
    roll_vol    = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        roll_tp_vol[i] = np.sum(tp_vol[i - lookback + 1 : i + 1])
        roll_vol[i]    = np.sum(volume[i - lookback + 1 : i + 1])

    vwap = np.where(roll_vol > 0, roll_tp_vol / roll_vol, typical)

    # ── 2. Micro trendline via LINEARREG (short linear regression of closes) ───
    micro_trendline = talib.LINEARREG(close, timeperiod=lookback)

    # Trendline slope: positive = uptrend, negative = downtrend
    trendline_slope = talib.LINEARREG_SLOPE(close, timeperiod=lookback)

    # ── 3. Short-term trend filter (fast EMA) ─────────────────────────────────
    fast_ema = talib.EMA(close, timeperiod=5)
    slow_ema = talib.EMA(close, timeperiod=8)

    # ── 4. ATR for proximity threshold ────────────────────────────────────────
    atr = talib.ATR(high, low, close, timeperiod=5)

    # ── 5. Proximity to VWAP / trendline ─────────────────────────────────────
    # "near" = close within 0.5 * ATR of vwap or micro_trendline
    half_atr          = 0.5 * atr
    near_vwap         = np.abs(close - vwap) <= half_atr
    near_trendline    = np.abs(close - micro_trendline) <= half_atr
    in_pullback_zone  = near_vwap | near_trendline

    # ── 6. Pullback detection over confirm_bars window ────────────────────────
    # Bullish pullback: uptrend (fast > slow), price dipped to zone recently
    # Bearish pullback: downtrend (fast < slow), price rose to zone recently

    uptrend   = fast_ema > slow_ema
    downtrend = fast_ema < slow_ema

    # Rolling check: was price in pullback zone within last confirm_bars bars?
    was_in_zone = np.zeros(n, dtype=bool)
    for i in range(confirm_bars, n):
        was_in_zone[i] = np.any(in_pullback_zone[i - confirm_bars : i])

    # ── 7. Bounce confirmation ────────────────────────────────────────────────
    # Bull bounce: uptrend, was near zone, now close > vwap AND close > trendline
    #              AND current bar is bullish (close > open)
    # Bear bounce: downtrend, was near zone, now close < vwap AND close < trendline
    #              AND current bar is bearish (close < open)

    bullish_bar   = close > open_
    bearish_bar   = close < open_

    bull_raw = (
        uptrend
        & was_in_zone
        & (close > vwap)
        & (close > micro_trendline)
        & bullish_bar
        & (trendline_slope > 0)
    )

    bear_raw = (
        downtrend
        & was_in_zone
        & (close < vwap)
        & (close < micro_trendline)
        & bearish_bar
        & (trendline_slope < 0)
    )

    # ── 8. Helper: numpy-level 1-bar lag ──────────────────────────────────────
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr        = np.empty(len(arr), dtype=bool)
        out_arr[0]     = False
        out_arr[1:]    = arr[:-1]
        return out_arr

    # ── 9. Assign outputs (all shift(1)) ──────────────────────────────────────
    idx = out.index

    out["vwap_micro_trendline_bounce_vwap"]          = pd.Series(vwap,             index=idx).shift(1)
    out["vwap_micro_trendline_bounce_trendline"]     = pd.Series(micro_trendline,  index=idx).shift(1)
    out["vwap_micro_trendline_bounce_slope"]         = pd.Series(trendline_slope,  index=idx).shift(1)
    out["vwap_micro_trendline_bounce_near_vwap"]     = pd.Series(_lag1(near_vwap),        index=idx, dtype=bool)
    out["vwap_micro_trendline_bounce_near_trend"]    = pd.Series(_lag1(near_trendline),   index=idx, dtype=bool)
    out["vwap_micro_trendline_bounce_in_zone"]       = pd.Series(_lag1(in_pullback_zone), index=idx, dtype=bool)
    out["vwap_micro_trendline_bounce_bull"]          = pd.Series(_lag1(bull_raw),         index=idx, dtype=bool)
    out["vwap_micro_trendline_bounce_bear"]          = pd.Series(_lag1(bear_raw),         index=idx, dtype=bool)

    return out