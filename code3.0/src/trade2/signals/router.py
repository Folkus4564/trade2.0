"""
signals/router.py - Strategy router: classifies bars by regime and delegates
to specialized sub-strategies (trend, range, volatile).

Usage (multi-TF mode - regime already in df):
    df = forward_fill_1h_regime(df_5m, ...)
    signals = route_signals(df, config)

Usage (single-TF mode):
    signals = route_signals(df, config,
                            hmm_labels=labels, hmm_bull_prob=bull,
                            hmm_bear_prob=bear, hmm_index=idx)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from trade2.signals.generator import _session_mask, _is_5m_data
from trade2.signals.strategies.trend    import trend_strategy
from trade2.signals.strategies.range    import range_strategy
from trade2.signals.strategies.volatile import volatile_strategy
from trade2.signals.strategies.cdc      import cdc_strategy


def apply_tv_signal_filter(sig_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Gate signal_long/signal_short using TV indicator _bull/_bear columns (signal_filter/both mode)."""
    tv_cfg = config.get("tv_indicators", {})
    tv_filter_long  = pd.Series(True, index=sig_df.index)
    tv_filter_short = pd.Series(True, index=sig_df.index)
    for tv_name, tv_ind_cfg in tv_cfg.items():
        if not isinstance(tv_ind_cfg, dict) or not tv_ind_cfg.get("enabled", False):
            continue
        if tv_ind_cfg.get("integration_mode", "hmm") not in ("signal_filter", "both"):
            continue
        bull_col = f"{tv_name}_bull"
        bear_col = f"{tv_name}_bear"
        if bull_col in sig_df.columns:
            tv_filter_long  = tv_filter_long  & sig_df[bull_col].astype(bool)
        if bear_col in sig_df.columns:
            tv_filter_short = tv_filter_short & sig_df[bear_col].astype(bool)
    out = sig_df.copy()
    out["signal_long"]  = (out["signal_long"].astype(bool)  & tv_filter_long).astype(int)
    out["signal_short"] = (out["signal_short"].astype(bool) & tv_filter_short).astype(int)
    return out


def ffill_tv_cols_to_5m(sig_df: pd.DataFrame, reg_feat: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill TV indicator _bull/_bear columns from 1H reg_feat into 5M sig_df."""
    tv_bb_cols = [c for c in reg_feat.columns
                  if (c.endswith("_bull") or c.endswith("_bear")) and not c.startswith("hmm_")]
    if not tv_bb_cols:
        return sig_df
    out = sig_df.copy()
    for col in tv_bb_cols:
        out[col] = reg_feat[col].reindex(out.index).ffill().fillna(False)
    return out


def _empty_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with all signal/exit columns zeroed out (disabled strategy contributes nothing)."""
    out = df.copy()
    for col in ("signal_long", "signal_short", "exit_long", "exit_short"):
        out[col] = 0
    for col in ("position_size_long", "position_size_short"):
        out[col] = 0.0
    out["signal_source"] = ""
    return out


def route_signals(
    df: pd.DataFrame,
    config: Dict[str, Any],
    # Single-TF mode: pass these explicitly
    hmm_labels: pd.Series = None,
    hmm_bull_prob: np.ndarray = None,
    hmm_bear_prob: np.ndarray = None,
    hmm_index: pd.Index = None,
    # Optional: pass trained model for self-transition probability gate (C1)
    hmm_model=None,
) -> pd.DataFrame:
    """
    Classify each bar by regime probability and delegate to the appropriate
    sub-strategy. Masks are mutually exclusive (priority: trend > range > volatile > cdc).

    Returns df with signal_long, signal_short, exit_long, exit_short,
    position_size_long, position_size_short, signal_source columns.
    """
    strat_cfg    = config["strategies"]
    trend_cfg    = strat_cfg["trend"]
    range_cfg    = strat_cfg["range"]
    volatile_cfg = strat_cfg["volatile"]
    cdc_cfg      = strat_cfg["cdc"]
    reg_cfg      = config["regime"]
    hmm_cfg      = config["hmm"]

    out = df.copy()

    # ---- 1. Align regime columns ----
    if "regime" not in out.columns:
        if hmm_labels is None:
            raise ValueError("Either pass regime in df (multi_tf) or provide hmm_labels (single_tf)")
        regime_s    = pd.Series(hmm_labels.values, index=hmm_index, name="regime")
        bull_prob_s = pd.Series(hmm_bull_prob,     index=hmm_index, name="bull_prob")
        bear_prob_s = pd.Series(hmm_bear_prob,     index=hmm_index, name="bear_prob")
        out["regime"]    = regime_s.reindex(out.index).fillna("sideways")
        out["bull_prob"] = bull_prob_s.reindex(out.index).fillna(0.0)
        out["bear_prob"] = bear_prob_s.reindex(out.index).fillna(0.0)

    if "sideways_prob" not in out.columns:
        out["sideways_prob"] = (1.0 - out["bull_prob"] - out["bear_prob"]).clip(lower=0.0)

    # ---- 2. Apply global transition cooldown ----
    cooldown = reg_cfg["transition_cooldown_bars"]
    if cooldown > 0:
        freq_mult     = 12 if _is_5m_data(out) else 1
        cooldown_bars = cooldown * freq_mult
        regime_changed = out["regime"] != out["regime"].shift(1)
        in_cooldown    = regime_changed.rolling(cooldown_bars, min_periods=1).sum() > 0
    else:
        in_cooldown = pd.Series(False, index=out.index)

    # ---- C1: Transition probability gate ----
    # Suppress entries when the HMM's own learned dynamics say the regime is unstable.
    # If bull self-transition prob < min_self_transition_prob, suppress all bull entries.
    min_self_tp = hmm_cfg.get("min_self_transition_prob", 0.0)
    bull_tp_ok  = True
    bear_tp_ok  = True
    if hmm_model is not None and min_self_tp > 0.0:
        try:
            bull_tp = hmm_model.self_transition_prob("bull")
            bear_tp = hmm_model.self_transition_prob("bear")
            bull_tp_ok = bull_tp >= min_self_tp
            bear_tp_ok = bear_tp >= min_self_tp
            if hmm_cfg.get("log_transition_matrix", False):
                try:
                    side_tp = hmm_model.self_transition_prob("sideways")
                    print(f"  [hmm-C1] self-transition: bull={bull_tp:.3f} | bear={bear_tp:.3f} | sideways={side_tp:.3f}")
                except Exception:
                    print(f"  [hmm-C1] self-transition: bull={bull_tp:.3f} | bear={bear_tp:.3f} (2-state: no sideways)")
                print(f"  [hmm-C1] gate threshold={min_self_tp:.3f} | bull_ok={bull_tp_ok} | bear_ok={bear_tp_ok}")
        except Exception as e:
            print(f"  [hmm-C1] self_transition_prob error: {e} -- gate skipped")

    # ---- 3. Classify bars (priority: trend > range > volatile) ----
    # Improvement #3: Use separate entry probability threshold (hysteresis).
    # min_prob_entry can be set higher than min_prob_exit for more selective entries.
    entry_prob = hmm_cfg.get("min_prob_entry", trend_cfg["min_prob"])
    bull_raw  = (out["bull_prob"]     >= entry_prob)             & ~in_cooldown & bull_tp_ok
    bear_raw  = (out["bear_prob"]     >= entry_prob)             & ~in_cooldown & bear_tp_ok
    range_raw = (out["sideways_prob"] >= range_cfg["min_prob"])  & ~in_cooldown

    # ---- C2: Drawdown suppression (Improvement #2) ----
    # Model-agnostic circuit breaker: suppress longs during sustained drawdowns.
    dd_cfg = config.get("drawdown_filter", {})
    if dd_cfg.get("enabled", False) and "High" in out.columns:
        freq_mult_dd  = 12 if _is_5m_data(out) else 1
        lookback_bars = dd_cfg.get("lookback_bars", 120) * freq_mult_dd
        rolling_high  = out["High"].rolling(lookback_bars, min_periods=1).max()
        dd            = (out["Close"] - rolling_high) / rolling_high.clip(lower=1e-9)
        suppress_long_dd  = dd_cfg.get("suppress_long_dd",    -0.05)
        suppress_short_rally = dd_cfg.get("suppress_short_rally", 0.05)
        suppress_long  = dd < suppress_long_dd
        suppress_short = dd > suppress_short_rally
        bull_raw = bull_raw & ~suppress_long
        bear_raw = bear_raw & ~suppress_short
        n_suppressed = int(suppress_long.sum() + suppress_short.sum())
        if n_suppressed > 0:
            print(f"  [dd-filter] suppressed {n_suppressed} bars (long={int(suppress_long.sum())} | short={int(suppress_short.sum())})")

    # ---- A2: Probability momentum filter ----
    # Suppress bull entries when bull_prob is NOT higher than it was N regime-TF bars ago.
    # "Momentum" = prob is still rising vs N bars back (directional check, not strictly consecutive).
    # In multi-TF mode (5M signals, 4H regime), bull_prob is forward-filled so we
    # shift by N * (signal bars per regime bar) to look back N regime bars.
    # prob_momentum_bars: number of regime-TF bars to look back (0 = disabled).
    prob_momentum_bars = hmm_cfg.get("prob_momentum_bars", 0)
    if prob_momentum_bars > 0 and "bull_prob" in out.columns:
        regime_tf_str = config.get("strategy", {}).get("regime_timeframe", "1H")
        sig_tf_str    = config.get("strategy", {}).get("signal_timeframe", "5M")
        mode_str      = config.get("strategy", {}).get("mode", "single_tf")
        _SIG_BARS_PER_H = {"5M": 12.0, "15M": 4.0, "30M": 2.0, "1H": 1.0, "4H": 0.25}
        if mode_str == "multi_tf":
            sig_per_regime = _SIG_BARS_PER_H.get(sig_tf_str, 12.0) / _SIG_BARS_PER_H.get(regime_tf_str, 1.0)
        else:
            sig_per_regime = 1.0
        pm_shift = int(prob_momentum_bars * sig_per_regime)
        prob_series = out["bull_prob"]
        # Momentum condition: current bull_prob strictly greater than N regime-bars ago
        bull_prob_n_ago = prob_series.shift(pm_shift)
        bull_momentum_ok = prob_series > bull_prob_n_ago
        bull_raw = bull_raw & bull_momentum_ok.fillna(False)
        n_pm_filtered = int((~bull_momentum_ok.fillna(False) & (out["bull_prob"] >= entry_prob)).sum())
        if n_pm_filtered > 0:
            print(f"  [prob-momentum] filtered {n_pm_filtered} bars (bull_prob not > value {prob_momentum_bars} bars ago)")

    # ---- C3: Regime freshness filter (Improvement #4) ----
    # Suppress entries when the regime has lasted far longer than the HMM expects.
    max_freshness = reg_cfg.get("max_regime_freshness", 0.0)
    if max_freshness > 0.0 and hmm_model is not None and "regime" in out.columns:
        decay_start = reg_cfg.get("freshness_decay_start", 1.0)
        # Convert signal bars to regime-TF bars
        regime_tf_str = config.get("strategy", {}).get("regime_timeframe", "1H")
        _SIG_BARS_PER_H = {"5M": 12.0, "15M": 4.0, "30M": 2.0, "1H": 1.0, "4H": 0.25}
        sig_tf_str = config.get("strategy", {}).get("signal_timeframe", "5M")
        mode_str   = config.get("strategy", {}).get("mode", "single_tf")
        if mode_str == "multi_tf":
            sig_per_regime = (_SIG_BARS_PER_H.get(sig_tf_str, 12.0) /
                              _SIG_BARS_PER_H.get(regime_tf_str, 1.0))
        else:
            sig_per_regime = 1.0
        # Count consecutive signal bars in same regime, then convert to regime-TF bars
        regime_col  = out["regime"]
        runs        = (regime_col != regime_col.shift(1)).cumsum()
        bars_signal = runs.groupby(runs).cumcount() + 1
        bars_signal = pd.Series(bars_signal.values, index=out.index)
        bars_regime_tf = bars_signal / max(sig_per_regime, 1.0)
        # Expected duration per regime from HMM self-transition probabilities
        def _expected_dur_bars(r):
            try:
                stp = hmm_model.self_transition_prob(r)
                return max(1.0 / (1.0 - stp), 1.0)
            except Exception:
                return 20.0
        expected_dur = regime_col.map(_expected_dur_bars)
        freshness    = bars_regime_tf / expected_dur.clip(lower=1.0)
        suppress_stale = freshness > max_freshness
        bull_raw = bull_raw & ~suppress_stale
        bear_raw = bear_raw & ~suppress_stale
        n_stale = int(suppress_stale.sum())
        if n_stale > 0:
            print(f"  [freshness] suppressed {n_stale} stale-regime bars (max_freshness={max_freshness})")

    max_prob = out[["bull_prob", "bear_prob", "sideways_prob"]].max(axis=1)
    vol_raw  = (max_prob < volatile_cfg["max_confidence"]) & ~in_cooldown

    trend_long  = bull_raw
    trend_short = bear_raw
    is_trend    = trend_long | trend_short
    range_mask  = range_raw & ~is_trend
    vol_mask    = vol_raw   & ~is_trend & ~range_mask

    # ---- 4. Call sub-strategies (with enabled guard) ----
    trend_enabled   = trend_cfg.get("enabled", True)
    range_enabled   = range_cfg.get("enabled", True)
    vol_enabled     = volatile_cfg.get("enabled", True)
    cdc_enabled     = cdc_cfg.get("enabled", False)

    trend_df   = trend_strategy(out,   config, trend_long, trend_short) if trend_enabled   else _empty_signals(out)
    range_df   = range_strategy(out,   config, range_mask)               if range_enabled   else _empty_signals(out)
    vol_df     = volatile_strategy(out, config, vol_mask)                 if vol_enabled     else _empty_signals(out)

    # CDC: regime_gated -> use trend masks; standalone -> all-True mask
    if cdc_enabled:
        if cdc_cfg["regime_gated"]:
            cdc_long_mask  = trend_long
            cdc_short_mask = trend_short
        else:
            cdc_long_mask  = pd.Series(True, index=out.index)
            cdc_short_mask = pd.Series(True, index=out.index)
        cdc_df = cdc_strategy(out, config, cdc_long_mask, cdc_short_mask)
    else:
        cdc_df = _empty_signals(out)

    # ---- 5. Merge signals ----
    result = out.copy()
    result["signal_long"]  = (
        (trend_df["signal_long"]  == 1) |
        (range_df["signal_long"]  == 1) |
        (vol_df["signal_long"]    == 1) |
        (cdc_df["signal_long"]    == 1)
    ).astype(int)

    result["signal_short"] = (
        (trend_df["signal_short"] == 1) |
        (range_df["signal_short"] == 1) |
        (vol_df["signal_short"]   == 1) |
        (cdc_df["signal_short"]   == 1)
    ).astype(int)

    # OR merge for exits: any active strategy wanting to exit triggers exit.
    # Disabled strategies return 0 (no vote), so they never block or force exits.
    result["exit_long"]  = (
        (trend_df["exit_long"]  == 1) |
        (range_df["exit_long"]  == 1) |
        (vol_df["exit_long"]    == 1) |
        (cdc_df["exit_long"]    == 1)
    ).astype(int)

    result["exit_short"] = (
        (trend_df["exit_short"] == 1) |
        (range_df["exit_short"] == 1) |
        (vol_df["exit_short"]   == 1) |
        (cdc_df["exit_short"]   == 1)
    ).astype(int)

    result["position_size_long"] = (
        trend_df["position_size_long"]  +
        range_df["position_size_long"]  +
        vol_df["position_size_long"]    +
        cdc_df["position_size_long"]
    )
    result["position_size_short"] = (
        trend_df["position_size_short"] +
        range_df["position_size_short"] +
        vol_df["position_size_short"]   +
        cdc_df["position_size_short"]
    )

    def _merge_source(t, r, v, c):
        out_arr = np.where(t != "", t, np.where(r != "", r, np.where(v != "", v, c)))
        return pd.Series(out_arr, index=result.index)

    result["signal_source"] = _merge_source(
        trend_df["signal_source"].values,
        range_df["signal_source"].values,
        vol_df["signal_source"].values,
        cdc_df["signal_source"].values,
    )

    # ---- 6. Direction filter (long_only / short_only) ----
    strat_top = config.get("strategy", {})
    if strat_top.get("long_only", False):
        result["signal_short"]        = 0
        result["exit_short"]          = 0
        result["position_size_short"] = 0.0
    if strat_top.get("short_only", False):
        result["signal_long"]        = 0
        result["exit_long"]          = 0
        result["position_size_long"] = 0.0

    # ---- 7. Diagnostics ----
    n_trend   = int((result["signal_source"] == "trend").sum())
    n_range   = int((result["signal_source"] == "range").sum())
    n_vol     = int((result["signal_source"] == "volatile").sum())
    n_cdc     = int((result["signal_source"] == "cdc").sum())
    n_total   = n_trend + n_range + n_vol + n_cdc
    print(f"  [router] signals: trend={n_trend} | range={n_range} | volatile={n_vol} | cdc={n_cdc} | total={n_total}")

    return result
