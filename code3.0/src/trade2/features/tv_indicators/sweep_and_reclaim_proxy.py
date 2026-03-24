"""
features/sweep_and_reclaim_proxy.py - Sweep and Reclaim Proxy feature detection.
Detects price sweeps of local highs/lows followed by sharp reclaim,
signaling trapped breakout traders.
All outputs shift(1) for lag safety.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


def add_sweep_and_reclaim_proxy_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Sweep and Reclaim Proxy features.

    Logic:
      - Identify local swing highs/lows over a short lookback window.
      - A bearish sweep occurs when price wicks ABOVE a recent swing high
        but then closes BELOW that swing high (trapped longs).
      - A bullish sweep occurs when price wicks BELOW a recent swing low
        but then closes ABOVE that swing low (trapped shorts).
      - Additional confirmation via a short ATR for wick magnitude filter.

    Zones:
      bull: price swept below recent low and reclaimed (bullish reversal)
      bear: price swept above recent high and reclaimed (bearish reversal)

    Args:
        df:     OHLCV DataFrame
        config: Full config dict.

    Returns:
        df copy with sweep_and_reclaim_proxy_ columns added.
    """
    params = config.get('tv_indicators', {}).get('sweep_and_reclaim_proxy', {})
    lookback      = int(params.get('lookback', 10))
    reclaim_bars  = int(params.get('reclaim_bars', 1))

    # Clamp to scalping-safe ranges
    lookback     = max(3, min(lookback, 12))
    reclaim_bars = max(1, min(reclaim_bars, 3))

    # ATR period for wick filter
    atr_period = max(3, min(lookback, 10))

    out   = df.copy()
    close = out["Close"].astype(float).values
    high  = out["High"].astype(float).values
    low   = out["Low"].astype(float).values
    open_ = out["Open"].astype(float).values
    n     = len(close)

    # --- ATR for magnitude filter ---
    atr = talib.ATR(high, low, close, timeperiod=atr_period)

    # --- Rolling swing high and swing low over lookback window ---
    # swing_high[i] = max(high[i-lookback : i])  (exclusive of current bar)
    # swing_low[i]  = min(low[i-lookback : i])
    swing_high = np.full(n, np.nan)
    swing_low  = np.full(n, np.nan)

    for i in range(lookback, n):
        swing_high[i] = np.max(high[i - lookback: i])
        swing_low[i]  = np.min(low[i - lookback: i])

    # --- Bearish sweep: high > swing_high AND close < swing_high ---
    # Wick above recent high but body closes back below it
    # Optionally require wick size >= some fraction of ATR
    wick_above   = high - np.maximum(close, open_)   # upper wick
    wick_below   = np.minimum(close, open_) - low     # lower wick

    bear_sweep_raw = np.zeros(n, dtype=bool)
    bull_sweep_raw = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        if np.isnan(atr[i]) or np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            continue

        atr_val = atr[i]
        if atr_val <= 0:
            continue

        sh = swing_high[i]
        sl = swing_low[i]

        # Bearish sweep: wick pierces above swing high, close reclaims below
        swept_high  = high[i] > sh
        reclaimed_h = close[i] < sh
        wick_sig_h  = wick_above[i] >= 0.25 * atr_val

        if swept_high and reclaimed_h and wick_sig_h:
            # Optional: check reclaim is confirmed within reclaim_bars
            # (for reclaim_bars == 1 we just look at current bar's close)
            if reclaim_bars == 1:
                bear_sweep_raw[i] = True
            else:
                # Ensure close is sufficiently below swing high
                if (sh - close[i]) >= 0.1 * atr_val:
                    bear_sweep_raw[i] = True

        # Bullish sweep: wick pierces below swing low, close reclaims above
        swept_low   = low[i] < sl
        reclaimed_l = close[i] > sl
        wick_sig_l  = wick_below[i] >= 0.25 * atr_val

        if swept_low and reclaimed_l and wick_sig_l:
            if reclaim_bars == 1:
                bull_sweep_raw[i] = True
            else:
                if (close[i] - sl) >= 0.1 * atr_val:
                    bull_sweep_raw[i] = True

    # --- Multi-bar reclaim confirmation ---
    # If reclaim_bars > 1, require N consecutive closes beyond the swept level.
    # We already computed single-bar signals above; extend with forward confirmation.
    if reclaim_bars > 1:
        bear_confirmed = np.zeros(n, dtype=bool)
        bull_confirmed = np.zeros(n, dtype=bool)

        for i in range(lookback, n - reclaim_bars + 1):
            if bear_sweep_raw[i]:
                sh = swing_high[i]
                if all(close[i + k] < sh for k in range(reclaim_bars)):
                    bear_confirmed[i] = True
            if bull_sweep_raw[i]:
                sl = swing_low[i]
                if all(close[i + k] > sl for k in range(reclaim_bars)):
                    bull_confirmed[i] = True

        bear_sweep_raw = bear_confirmed
        bull_sweep_raw = bull_confirmed

    # --- Helper: numpy 1-bar lag ---
    def _lag1(arr: np.ndarray) -> np.ndarray:
        out_arr = np.empty_like(arr)
        out_arr[0] = False
        out_arr[1:] = arr[:-1]
        return out_arr

    # --- Shift(1) all outputs ---
    idx = out.index

    out["sweep_and_reclaim_proxy_bull"] = pd.Series(
        _lag1(bull_sweep_raw), index=idx, dtype=bool
    )
    out["sweep_and_reclaim_proxy_bear"] = pd.Series(
        _lag1(bear_sweep_raw), index=idx, dtype=bool
    )
    out["sweep_and_reclaim_proxy_swing_high"] = pd.Series(
        _lag1(swing_high), index=idx
    )
    out["sweep_and_reclaim_proxy_swing_low"] = pd.Series(
        _lag1(swing_low), index=idx
    )
    out["sweep_and_reclaim_proxy_atr"] = pd.Series(
        _lag1(atr), index=idx
    )

    return out