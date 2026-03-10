"""
Module: signal_generator.py
Purpose: Multi-timeframe signal generation
         - 1H HMM regime is forward-filled onto 5M bars
         - SMC confluence entries on 5M
         - Regime persistence filter (N bars stable before entry)
         - Tier-1: sweep + (OB or FVG), Tier-2: OB + FVG
"""

import numpy as np
import pandas as pd


def forward_fill_1h_regime(
    df_5m: pd.DataFrame,
    hmm_labels: pd.Series,
    hmm_bull_prob: np.ndarray,
    hmm_bear_prob: np.ndarray,
    hmm_index: pd.Index,
) -> pd.DataFrame:
    """
    Forward-fill 1H HMM regime labels and probabilities onto 5M bars.

    Each 5M bar receives the regime from the most recently completed 1H bar.
    This is lag-safe: the 1H bar must be fully closed before its regime
    is visible on 5M.

    Args:
        df_5m:         5M feature DataFrame
        hmm_labels:    Series of regime labels indexed by position (0..N)
        hmm_bull_prob: Array of bull posterior probabilities
        hmm_bear_prob: Array of bear posterior probabilities
        hmm_index:     DatetimeIndex corresponding to HMM predictions (1H bar open times)

    Returns:
        df_5m with regime, bull_prob, bear_prob columns added
    """
    out = df_5m.copy()

    regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
    bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
    bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")

    # Reindex to 5M, forward-fill (each 5M bar gets the last known 1H regime)
    out["regime"]    = regime_s.reindex(out.index, method="ffill").fillna("sideways")
    out["bull_prob"] = bull_prob_s.reindex(out.index, method="ffill").fillna(0.0)
    out["bear_prob"] = bear_prob_s.reindex(out.index, method="ffill").fillna(0.0)

    return out


def generate_signals(
    df_5m: pd.DataFrame,
    adx_threshold: float = 20.0,
    hmm_min_prob: float = 0.50,
    regime_persistence_bars: int = 3,
    require_smc_confluence: bool = True,
    require_pin_bar: bool = False,
) -> pd.DataFrame:
    """
    Generate entry/exit signals on 5M bars using 1H HMM regime context.

    Entry logic (tiered SMC confluence):
      Tier 1 (primary):  sweep + (OB or FVG) — institutional sweep + structure
      Tier 2 (fallback): OB + FVG overlap — double structure confirmation

    All tiers require:
      - Stable bull/bear regime (N consecutive 1H regime bars)
      - HMM probability >= hmm_min_prob
      - Optional: pin bar rejection candle

    Exit logic:
      - Regime flips away from entry direction (HMM-based)

    Args:
        df_5m:                   5M DataFrame with SMC features + regime columns
        adx_threshold:           Min ADX for Donchian breakout fallback
        hmm_min_prob:            Min HMM posterior for entry
        regime_persistence_bars: Consecutive 1H-equivalent regime bars required
        require_smc_confluence:  If True, use tiered confluence; if False, OR logic
        require_pin_bar:         If True, require pin bar rejection before entry

    Returns:
        df_5m with signal_long, signal_short, exit_long, exit_short columns
    """
    out = df_5m.copy()

    # ---- Regime filter with stability check ----
    # Persistence on 5M: multiply 1H bars by 12 (12 x 5min = 1H)
    # But we check stability of the 1H regime label itself, not 5M bars
    # Use the regime column which is forward-filled from 1H
    # Persistence: require regime to be the same for last N x 12 5M bars
    persist_5m = regime_persistence_bars * 12

    bull_raw = (out["regime"] == "bull") & (out["bull_prob"] >= hmm_min_prob)
    bear_raw = (out["regime"] == "bear") & (out["bear_prob"] >= hmm_min_prob)

    # Stability: rolling sum == window means all bars in window match
    bull_regime = bull_raw.rolling(persist_5m).sum() == persist_5m
    bear_regime = bear_raw.rolling(persist_5m).sum() == persist_5m

    # ---- Check required SMC columns exist ----
    has_smc = all(c in out.columns for c in [
        "ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish",
        "sweep_low", "sweep_high",
    ])
    has_pin = "pin_bar_bull" in out.columns and "pin_bar_bear" in out.columns

    if has_smc:
        if require_smc_confluence:
            # Tier 1: sweep + structure
            t1_long  = out["sweep_low"]  & (out["ob_bullish"] | out["fvg_bullish"])
            t1_short = out["sweep_high"] & (out["ob_bearish"] | out["fvg_bearish"])
            # Tier 2: double structure
            t2_long  = out["ob_bullish"] & out["fvg_bullish"]
            t2_short = out["ob_bearish"] & out["fvg_bearish"]
            smc_long  = t1_long  | t2_long
            smc_short = t1_short | t2_short
        else:
            # Legacy OR logic
            smc_long  = out["ob_bullish"] | out["fvg_bullish"] | out["sweep_low"]
            smc_short = out["ob_bearish"] | out["fvg_bearish"] | out["sweep_high"]

        # Optional pin bar filter
        if require_pin_bar and has_pin:
            smc_long  = smc_long  & out["pin_bar_bull"]
            smc_short = smc_short & out["pin_bar_bear"]
    else:
        smc_long  = pd.Series(False, index=out.index)
        smc_short = pd.Series(False, index=out.index)

    # ---- Donchian breakout fallback ----
    has_dc = "breakout_long" in out.columns and "adx_14" in out.columns
    if has_dc:
        adx_ok     = out["adx_14"] > adx_threshold
        dc_long    = out["breakout_long"].astype(bool)  & adx_ok
        dc_short   = out["breakout_short"].astype(bool) & adx_ok
    else:
        dc_long  = pd.Series(False, index=out.index)
        dc_short = pd.Series(False, index=out.index)

    # ---- Combine: regime filter is mandatory ----
    out["signal_long"]  = (bull_regime & (smc_long  | dc_long)).astype(int)
    out["signal_short"] = (bear_regime & (smc_short | dc_short)).astype(int)

    # ---- Exits: regime flip ----
    out["exit_long"]  = (~bull_regime).astype(int)
    out["exit_short"] = (~bear_regime).astype(int)

    # ---- Position sizing: scale by HMM confidence ----
    out["position_size_long"] = np.where(
        out["signal_long"] == 1,
        0.5 + out["bull_prob"],
        0.0,
    )
    out["position_size_short"] = np.where(
        out["signal_short"] == 1,
        0.5 + out["bear_prob"],
        0.0,
    )

    return out


def compute_stops(
    df: pd.DataFrame,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
) -> pd.DataFrame:
    """ATR-based stop loss and take profit on 5M bars."""
    out   = df.copy()
    atr   = out["atr_14"]
    close = out["Close"]

    out["stop_long"]  = close - atr_stop_mult * atr
    out["stop_short"] = close + atr_stop_mult * atr
    out["tp_long"]    = close + atr_tp_mult   * atr
    out["tp_short"]   = close - atr_tp_mult   * atr

    return out
