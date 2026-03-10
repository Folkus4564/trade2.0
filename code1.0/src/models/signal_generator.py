"""
Module: signal_generator.py
Purpose: Convert features and HMM regime into buy/sell signals for XAUUSD strategy
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from typing import Tuple

from src.config import get_config


# ── Signal generation ─────────────────────────────────────────────────────────

_LONDON_HOURS = set(range(7, 17))   # 07-16 UTC
_NY_HOURS     = set(range(13, 22))  # 13-21 UTC
_ACTIVE_HOURS = _LONDON_HOURS | _NY_HOURS  # London + NY (most liquid)


def _session_mask(index: pd.DatetimeIndex, allowed_hours: set) -> pd.Series:
    """Return boolean mask True for bars in allowed UTC hours."""
    if index.tz is None:
        hours = index.hour
    else:
        hours = index.tz_convert("UTC").hour
    return pd.Series(hours.isin(allowed_hours), index=index)


def generate_signals(
    df: pd.DataFrame,
    hmm_labels: pd.Series,
    hmm_bull_prob: np.ndarray,
    hmm_bear_prob: np.ndarray,
    hmm_index: pd.Index,
    adx_threshold: float = 20.0,
    hmm_min_prob: float = 0.55,
    atr_expansion_filter: bool = True,
    use_smc: bool = True,
    session_filter: bool = None,
) -> pd.DataFrame:
    """
    Generate long/short entry and exit signals using:
    1. SMC signals (primary): Order Blocks, Fair Value Gaps, Liquidity Sweeps
       filtered by HMM regime.
    2. Donchian Channel breakout signals (secondary/fallback) filtered by HMM.

    Entry rules (SMC + HMM):
    - LONG:  HMM regime = bull AND bull_prob >= hmm_min_prob
             AND (bullish_ob OR bullish_fvg OR sweep_low)
    - SHORT: HMM regime = bear AND bear_prob >= hmm_min_prob
             AND (bearish_ob OR bearish_fvg OR sweep_high)

    Entry rules (Breakout + HMM, used if use_smc=False or combined):
    - LONG:  Close breaks above N-bar Donchian upper AND bull regime AND ADX AND ATR
    - SHORT: Close breaks below N-bar Donchian lower AND bear regime AND ADX AND ATR

    Exit rules:
    - Long exit:  Close pulls back below Donchian midline OR HMM flips to bear/sideways
    - Short exit: Close rallies back above Donchian midline OR HMM flips to bull/sideways

    Args:
        df:                   Feature-enriched OHLCV DataFrame
        hmm_labels:           Series of regime labels ('bull', 'bear', 'sideways')
        hmm_bull_prob:        Array of bull posterior probabilities
        hmm_bear_prob:        Array of bear posterior probabilities
        hmm_index:            DatetimeIndex corresponding to HMM predictions
        adx_threshold:        Minimum ADX for trend confirmation (breakout signals)
        hmm_min_prob:         Minimum HMM posterior probability for entry
        atr_expansion_filter: If True, require ATR > ATR_MA for breakout signals
        use_smc:              If True, use SMC signals as primary entry (default: True)

    Returns:
        DataFrame with columns: signal_long, signal_short, exit_long, exit_short,
                                 position_size_long, position_size_short
    """
    # Read sizing and session from config
    cfg = get_config()
    hmm_cfg     = cfg.get("hmm", {})
    sess_cfg    = cfg.get("session", {})
    sizing_base = hmm_cfg.get("sizing_base", 0.50)
    sizing_max  = hmm_cfg.get("sizing_max",  1.50)
    if session_filter is None:
        session_filter = sess_cfg.get("enabled", False)
    allowed_hours = set(sess_cfg.get("allowed_hours_utc", list(_ACTIVE_HOURS)))

    out = df.copy()

    # Align HMM outputs to the full DataFrame index
    regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
    bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
    bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")

    out["regime"]    = regime_s.reindex(out.index).fillna("sideways")
    out["bull_prob"] = bull_prob_s.reindex(out.index).fillna(0.0)
    out["bear_prob"] = bear_prob_s.reindex(out.index).fillna(0.0)

    # ── HMM regime filter (SOFT gate) ────────────────────────────────────────
    # HMM is used for confidence-based position sizing, not as a hard binary blocker.
    # Trades are allowed in any non-sideways regime; size is scaled by HMM confidence.
    # Only sideways regime is excluded from new entries.
    bull_regime = (out["regime"] == "bull")
    bear_regime = (out["regime"] == "bear")

    # Hard probability gate is applied only when explicitly requested
    if hmm_min_prob > 0:
        bull_regime = bull_regime & (out["bull_prob"] >= hmm_min_prob)
        bear_regime = bear_regime & (out["bear_prob"] >= hmm_min_prob)

    # ── SMC entry signals ─────────────────────────────────────────────────────
    # Check if SMC feature columns exist (may not if add_smc_features was not called)
    has_smc = all(c in out.columns for c in ["ob_bullish", "ob_bearish",
                                              "fvg_bullish", "fvg_bearish",
                                              "sweep_low", "sweep_high"])

    if use_smc and has_smc:
        # Long: bull_regime AND any bullish SMC trigger
        smc_long  = bull_regime & (
            out["ob_bullish"] | out["fvg_bullish"] | out["sweep_low"]
        )
        # Short: bear_regime AND any bearish SMC trigger
        smc_short = bear_regime & (
            out["ob_bearish"] | out["fvg_bearish"] | out["sweep_high"]
        )
    else:
        smc_long  = pd.Series(False, index=out.index)
        smc_short = pd.Series(False, index=out.index)

    # ── Donchian breakout entry signals (secondary / standalone fallback) ─────
    broke_up   = out["breakout_long"]  == 1
    broke_down = out["breakout_short"] == 1
    adx_ok     = out["adx_14"] > adx_threshold
    atr_exp    = out["atr_expansion"] == 1 if atr_expansion_filter else pd.Series(True, index=out.index)

    dc_long  = broke_up   & adx_ok & atr_exp & bull_regime
    dc_short = broke_down & adx_ok & atr_exp & bear_regime

    # ── Session filter (optional) ─────────────────────────────────────────────
    # Only enter during London + NY sessions (7-21 UTC) — most liquid, tightest spreads.
    # Existing positions are NOT closed by session filter — only new entries are blocked.
    if session_filter:
        in_session = _session_mask(out.index, allowed_hours)
        smc_long   = smc_long  & in_session
        smc_short  = smc_short & in_session
        dc_long    = dc_long   & in_session
        dc_short   = dc_short  & in_session

    # ── Combine signals: SMC OR Donchian breakout ─────────────────────────────
    out["signal_long"]  = (smc_long  | dc_long).astype(int)
    out["signal_short"] = (smc_short | dc_short).astype(int)

    # ── Exit conditions ───────────────────────────────────────────────────────
    # Exit only on regime flip: let breakout trades run while HMM confirms trend.
    # dc_mid exits are intentionally NOT used — breakouts normally consolidate
    # before continuing, and dc_mid exits would terminate valid trades prematurely.
    # With 2-state HMM (bull/bear), sideways never appears — exits are purely regime flip.
    out["exit_long"]  = (~bull_regime).astype(int)
    out["exit_short"] = (~bear_regime).astype(int)

    # ── Position sizing: confidence-based scaling ─────────────────────────────
    # Size = 0.5 + regime_probability, capped at 1.5x
    # Low-confidence regime = smaller size; high-confidence = full size.
    # This replaces hard blocking with a graduated response to uncertainty.
    out["position_size_long"]  = np.where(
        out["signal_long"]  == 1,
        np.clip(sizing_base + out["bull_prob"], 0.1, sizing_max),
        0.0
    )
    out["position_size_short"] = np.where(
        out["signal_short"] == 1,
        np.clip(sizing_base + out["bear_prob"], 0.1, sizing_max),
        0.0
    )

    return out


def compute_stops(
    df: pd.DataFrame,
    atr_stop_mult: float = 2.0,
    atr_tp_mult: float   = 4.0,
) -> pd.DataFrame:
    """
    Compute ATR-based stop loss and take profit levels.

    Stop loss  = entry_price +/- atr_stop_mult * ATR
    Take profit = entry_price +/- atr_tp_mult  * ATR

    Args:
        df: Signal DataFrame with atr_14 column
        atr_stop_mult: ATR multiplier for stop loss
        atr_tp_mult:   ATR multiplier for take profit

    Returns:
        df with stop_long, stop_short, tp_long, tp_short columns added
    """
    out = df.copy()
    atr = out["atr_14"]
    close = out["Close"]

    out["stop_long"]   = close - atr_stop_mult * atr
    out["stop_short"]  = close + atr_stop_mult * atr
    out["tp_long"]     = close + atr_tp_mult   * atr
    out["tp_short"]    = close - atr_tp_mult   * atr

    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.loader import load_split
    from src.data.features import add_features, get_hmm_feature_matrix
    from src.models.hmm_model import XAUUSDRegimeModel

    train, val, test = load_split("1H")
    train_feat = add_features(train)
    X_train, idx_train = get_hmm_feature_matrix(train_feat)

    model = XAUUSDRegimeModel(n_states=3)
    model.fit(X_train)

    labels    = model.regime_labels(X_train)
    bull_prob = model.bull_probability(X_train)
    bear_prob = model.bear_probability(X_train)

    signals = generate_signals(train_feat, labels, bull_prob, bear_prob, idx_train)
    signals = compute_stops(signals)

    n_long  = signals["signal_long"].sum()
    n_short = signals["signal_short"].sum()
    print(f"Long signals:  {n_long}")
    print(f"Short signals: {n_short}")
    print(signals[["Close","regime","bull_prob","signal_long","signal_short","stop_long"]].tail(10))
