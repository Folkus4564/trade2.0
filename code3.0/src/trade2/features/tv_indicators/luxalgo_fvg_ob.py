"""
tv_indicators/luxalgo_fvg_ob.py

LuxAlgo Smart Money Concepts - Fair Value Gaps and Order Blocks.

Translated from Pine Script: 'Smart Money Concepts [LuxAlgo]'
Original: https://www.tradingview.com/script/HfFdTLAc/
License: CC BY-NC-SA 4.0

Adds FVG and OB features on top of existing SMC columns
(swing_high, swing_low, bos_bullish, bos_bearish, choch_bullish, choch_bearish)
which are computed by smc_luxalgo.py earlier in the pipeline.

All output columns are shift(1) for lag safety.

Output columns:
  fvg_bullish       bool  - bullish FVG formed on prior bar
  fvg_bearish       bool  - bearish FVG formed on prior bar
  in_bull_fvg       bool  - close is inside an unmitigated bullish FVG
  in_bear_fvg       bool  - close is inside an unmitigated bearish FVG
  price_in_bull_ob  bool  - close is inside an unmitigated bullish order block
  price_in_bear_ob  bool  - close is inside an unmitigated bearish order block
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

def _detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap detection (LuxAlgo logic).

    Bullish FVG at bar i: low[i] > high[i-2]  (gap between current low and 2-bars-ago high)
    Bearish FVG at bar i: high[i] < low[i-2]  (gap between current high and 2-bars-ago low)

    Active FVGs are tracked until mitigated:
      - Bullish FVG mitigated when low trades below FVG bottom (= high[i-2])
      - Bearish FVG mitigated when high trades above FVG top  (= low[i-2])
    """
    n          = len(df)
    high_v     = df["High"].values.astype(float)
    low_v      = df["Low"].values.astype(float)
    close_v    = df["Close"].values.astype(float)

    bull_fvg_raw = np.zeros(n, dtype=bool)
    bear_fvg_raw = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if low_v[i] > high_v[i - 2]:
            bull_fvg_raw[i] = True
        if high_v[i] < low_v[i - 2]:
            bear_fvg_raw[i] = True

    in_bull_fvg = np.zeros(n, dtype=bool)
    in_bear_fvg = np.zeros(n, dtype=bool)

    # Each entry: [top, bottom]
    active_bull: list = []
    active_bear: list = []

    for i in range(2, n):
        # Register new FVGs
        if bull_fvg_raw[i]:
            active_bull.append([low_v[i], high_v[i - 2]])   # top, bottom
        if bear_fvg_raw[i]:
            active_bear.append([low_v[i - 2], high_v[i]])   # top, bottom

        # Mitigate: bull FVG gone when price dips below its bottom
        active_bull = [fg for fg in active_bull if low_v[i] >= fg[1]]
        # Mitigate: bear FVG gone when price rises above its top
        active_bear = [fg for fg in active_bear if high_v[i] <= fg[0]]

        c = close_v[i]
        in_bull_fvg[i] = any(fg[1] <= c <= fg[0] for fg in active_bull)
        in_bear_fvg[i] = any(fg[1] <= c <= fg[0] for fg in active_bear)

    idx = df.index
    out = df.copy()
    # Shift(1) for lag safety
    out["fvg_bullish"] = pd.Series(
        np.concatenate([[False], bull_fvg_raw[:-1]]), index=idx, dtype=bool
    )
    out["fvg_bearish"] = pd.Series(
        np.concatenate([[False], bear_fvg_raw[:-1]]), index=idx, dtype=bool
    )
    out["in_bull_fvg"] = pd.Series(
        np.concatenate([[False], in_bull_fvg[:-1]]), index=idx, dtype=bool
    )
    out["in_bear_fvg"] = pd.Series(
        np.concatenate([[False], in_bear_fvg[:-1]]), index=idx, dtype=bool
    )
    return out


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Order Block detection (simplified LuxAlgo logic).

    Bullish OB (demand zone): the last bearish candle (close < open) before
    a bullish BOS or CHoCH. Price entering this zone signals institutional demand.

    Bearish OB (supply zone): the last bullish candle (close > open) before
    a bearish BOS or CHoCH. Price entering this zone signals institutional supply.

    OBs are mitigated when price closes beyond their far edge:
      - Bullish OB mitigated when close < ob_bottom
      - Bearish OB mitigated when close > ob_top

    Requires columns: bos_bullish, bos_bearish, choch_bullish, choch_bearish
    (computed by smc_luxalgo.py). Falls back to zeros if not present.
    """
    n       = len(df)
    open_v  = df["Open"].values.astype(float)
    high_v  = df["High"].values.astype(float)
    low_v   = df["Low"].values.astype(float)
    close_v = df["Close"].values.astype(float)

    def _get_bool_col(name: str) -> np.ndarray:
        if name in df.columns:
            return df[name].values.astype(bool)
        return np.zeros(n, dtype=bool)

    bos_bull   = _get_bool_col("bos_bullish")
    bos_bear   = _get_bool_col("bos_bearish")
    choch_bull = _get_bool_col("choch_bullish")
    choch_bear = _get_bool_col("choch_bearish")

    price_in_bull_ob = np.zeros(n, dtype=bool)
    price_in_bear_ob = np.zeros(n, dtype=bool)

    # Each entry: [top, bottom]
    bull_obs: list = []
    bear_obs: list = []

    for i in range(lookback, n):
        # Bullish break -> find last bearish candle in lookback window
        if bos_bull[i] or choch_bull[i]:
            for j in range(i - 1, max(i - lookback, -1), -1):
                if close_v[j] < open_v[j]:          # bearish candle
                    bull_obs.append([high_v[j], low_v[j]])
                    break

        # Bearish break -> find last bullish candle in lookback window
        if bos_bear[i] or choch_bear[i]:
            for j in range(i - 1, max(i - lookback, -1), -1):
                if close_v[j] > open_v[j]:          # bullish candle
                    bear_obs.append([high_v[j], low_v[j]])
                    break

        # Mitigate OBs
        bull_obs = [ob for ob in bull_obs if close_v[i] >= ob[1]]
        bear_obs = [ob for ob in bear_obs if close_v[i] <= ob[0]]

        c = close_v[i]
        price_in_bull_ob[i] = any(ob[1] <= c <= ob[0] for ob in bull_obs)
        price_in_bear_ob[i] = any(ob[1] <= c <= ob[0] for ob in bear_obs)

    idx = df.index
    out = df.copy()
    out["price_in_bull_ob"] = pd.Series(
        np.concatenate([[False], price_in_bull_ob[:-1]]), index=idx, dtype=bool
    )
    out["price_in_bear_ob"] = pd.Series(
        np.concatenate([[False], price_in_bear_ob[:-1]]), index=idx, dtype=bool
    )
    return out


# ---------------------------------------------------------------------------
# Entry point (required by tv_indicators registry)
# ---------------------------------------------------------------------------

def add_luxalgo_fvg_ob_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add LuxAlgo FVG and Order Block features to df.

    Config key: tv_indicators.luxalgo_fvg_ob
      enabled:  bool   (default False)
      ob_lookback: int (default 20) - bars to look back for OB candle
    """
    cfg = (config or {}).get("tv_indicators", {}).get("luxalgo_fvg_ob", {})
    if not cfg.get("enabled", False):
        return df

    ob_lookback = int(cfg.get("ob_lookback", 20))

    out = _detect_fvg(df)
    out = _detect_order_blocks(out, lookback=ob_lookback)
    return out
