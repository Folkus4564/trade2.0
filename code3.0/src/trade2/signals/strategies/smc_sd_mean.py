"""
signals/strategies/smc_sd_mean.py - SD Adaptive Mean Zone Reversal sub-strategy.

Philosophy:
  BUY when price is at a GENUINE EXTREME: sd_zone <= -min_entry_zone (e.g. -2 or -3).
  SELL when price is at a GENUINE EXTREME: sd_zone >= +min_entry_zone.

  The SD zone (-4..+4) uses dynamic ATR-scaled bands from the Pinescript indicator.
  Entering at -2 SD gives ~2:1 natural R:R to mean; at -3 SD gives ~3:1 R:R.
  This is where the ~70% WR on M1 comes from: extreme deviations almost always revert.

  SMC zones are OPTIONAL additional confirmation (not required by default).
  Works best on M1 timeframe with frequent extremes and fast mean reversion.

Entry logic:

  LONG:
    1. SD Zone: sd_zone <= -min_entry_zone  (price deeply below adaptive mean)
    2. HMM gate (optional): bull_prob >= min_prob OR disabled
    3. Wick confirmation (optional): bullish bar with lower wick
    4. SMC zone (optional): OB/demand zone for higher conviction
    5. Cooldown: no signal within cooldown_bars

  SHORT (mirror):
    1. SD Zone: sd_zone >= +min_entry_zone
    2. HMM gate (optional)
    3. Wick confirmation (optional)
    4. SMC zone (optional)
    5. Cooldown

Exits:
  SL: ATR-based stop (below entry for longs)
  TP: mean-reversion target = 0 SD line (configurable via atr_tp_mult or use_mean_tp)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def smc_sd_mean_strategy(
    df: pd.DataFrame,
    config: Dict[str, Any],
    long_mask: pd.Series,
    short_mask: pd.Series,
) -> pd.DataFrame:
    """
    SMC + SD Adaptive Mean Reversion sub-strategy.

    Args:
        df:         Signal DataFrame with SMC features, SD mean columns, HMM regime.
        config:     Full config dict.
        long_mask:  Boolean mask passed by router (all-True for reversal strategies).
        short_mask: Boolean mask passed by router (all-True for reversal strategies).

    Returns:
        DataFrame with 7 standard signal columns.
    """
    from trade2.signals.generator import _session_mask, _ACTIVE_HOURS

    cfg = config.get("strategies", {}).get("smc_sd_mean", {})
    out = df.copy()
    idx = out.index

    def _get_bool(col: str) -> pd.Series:
        if col in out.columns:
            return out[col].fillna(False).astype(bool)
        return pd.Series(False, index=idx)

    def _get_float(col: str, default: float = 0.0) -> pd.Series:
        if col in out.columns:
            return out[col].fillna(default).astype(float)
        return pd.Series(default, index=idx)

    # ================================================================
    # 1. SD ADAPTIVE MEAN ZONE FILTER (PRIMARY SIGNAL)
    #    Use sd_zone (-4..+4) for entry — dynamic ATR-scaled bands.
    #    Zone entry gives natural mean-reversion R:R:
    #      zone -2 -> TP at 0 = +2 SD gain, SL at -3 = -1 SD loss → 2:1 R:R
    #      zone -3 -> TP at 0 = +3 SD gain, SL at -4 = -1 SD loss → 3:1 R:R
    #    This is what drives the ~70% WR on M1 from the Pinescript.
    # ================================================================
    min_entry_zone = int(cfg.get("min_entry_zone", 2))   # enter when |sd_zone| >= this
    sd_smoothed = _get_float("sd_smoothed", default=0.0)
    sd_zone     = _get_float("sd_zone",     default=0.0).round().astype(int)

    sd_long_ok  = sd_zone <= -min_entry_zone
    sd_short_ok = sd_zone >=  min_entry_zone

    # Legacy threshold fallback (if sd_zone unavailable, use raw smoothed value)
    if sd_long_ok.sum() == 0 and sd_short_ok.sum() == 0:
        sd_entry_threshold = float(cfg.get("sd_entry_threshold", 0.5))
        sd_long_ok  = sd_smoothed <= -sd_entry_threshold
        sd_short_ok = sd_smoothed >=  sd_entry_threshold

    # ================================================================
    # 2. SMC ZONE DETECTION
    #    Primary: LuxAlgo structural OB retests (higher quality)
    #    Secondary: Demand/Supply zone retests
    # ================================================================
    struct_ob_bull = _get_bool("struct_ob_bull_retest")
    struct_ob_bear = _get_bool("struct_ob_bear_retest")
    dz_demand      = _get_bool("dz_demand_retest")
    dz_supply      = _get_bool("dz_supply_retest")

    # Fallback: impulse OBs if LuxAlgo not available
    impulse_ob_bull = _get_bool("ob_bullish")
    impulse_ob_bear = _get_bool("ob_bearish")

    # BB position as extra zone: near lower/upper Bollinger Band = statistical extreme zone
    # bb_pos near 0 = close near lower BB = demand zone; near 1 = close near upper BB = supply zone
    bb_zone_threshold = float(cfg.get("bb_zone_threshold", 0.25))
    bb_pos_col = _get_float("bb_pos", default=0.5)
    bb_zone_long  = bb_pos_col <= bb_zone_threshold
    bb_zone_short = bb_pos_col >= (1.0 - bb_zone_threshold)

    # require_smc_zone: if False (default), SD zone alone is sufficient — SMC is optional bonus
    require_smc_zone = cfg.get("require_smc_zone", False)
    use_struct_only  = cfg.get("use_struct_ob_only", False)
    use_bb_zone      = cfg.get("use_bb_zone", False)  # off by default — SD zone replaces BB

    if use_struct_only:
        zone_long  = struct_ob_bull | dz_demand
        zone_short = struct_ob_bear | dz_supply
    else:
        zone_long  = struct_ob_bull | dz_demand | impulse_ob_bull
        zone_short = struct_ob_bear | dz_supply | impulse_ob_bear

    if use_bb_zone:
        zone_long  = zone_long  | bb_zone_long
        zone_short = zone_short | bb_zone_short

    if not require_smc_zone:
        # SD zone is the primary filter — SMC zone not required
        zone_long  = pd.Series(True, index=idx)
        zone_short = pd.Series(True, index=idx)

    # ================================================================
    # 3. HMM REGIME GATE (optional)
    #    Standard (trend-aligned): bull_prob → longs, bear_prob → shorts
    #    Inverted (mean-reversion): bear_prob → longs, bull_prob → shorts
    #    SD zone is the primary quality filter — HMM adds directional confirmation.
    # ================================================================
    min_prob_long  = float(cfg.get("min_prob",       0.50))
    min_prob_short = float(cfg.get("min_prob_short", 0.50))
    regime_gated   = cfg.get("regime_gated", True)   # on by default — HMM improves WR

    has_hmm = "bear_prob" in out.columns and "bull_prob" in out.columns
    use_inverted_hmm = cfg.get("use_inverted_hmm", False)  # default: standard trend-aligned
    if regime_gated and has_hmm:
        if use_inverted_hmm:
            # Mean-reversion: bear state = price in discount = buy the dip
            hmm_long_ok  = out["bear_prob"].fillna(0.0) >= min_prob_long
            hmm_short_ok = out["bull_prob"].fillna(0.0) >= min_prob_short
        else:
            # Trend-aligned: buy SD extremes in uptrends, sell in downtrends
            # Bull trend dip to -2 SD = high-conviction pullback entry
            hmm_long_ok  = out["bull_prob"].fillna(0.0) >= min_prob_long
            hmm_short_ok = out["bear_prob"].fillna(0.0) >= min_prob_short
    else:
        # No HMM gate: SD zone alone is the filter
        hmm_long_ok  = pd.Series(True, index=idx)
        hmm_short_ok = pd.Series(True, index=idx)

    # Apply passed masks (router all-True for reversal strategies)
    hmm_long_ok  = hmm_long_ok  & long_mask.astype(bool)
    hmm_short_ok = hmm_short_ok & short_mask.astype(bool)

    # ================================================================
    # 4. REJECTION CANDLE CONFIRMATION
    #    Long: bullish bar with meaningful lower wick (rejection of lower prices)
    #    Short: bearish bar with meaningful upper wick (rejection of higher prices)
    # ================================================================
    require_wick = cfg.get("require_wick_confirmation", True)
    wick_ratio   = float(cfg.get("wick_ratio", 0.3))

    open_p  = out["Open"].astype(float)
    high_p  = out["High"].astype(float)
    low_p   = out["Low"].astype(float)
    close_p = out["Close"].astype(float)
    bar_range = (high_p - low_p).clip(lower=1e-10)

    if require_wick:
        # Long: bullish body + lower wick >= wick_ratio * range
        lower_wick  = (open_p - low_p).clip(lower=0.0)
        upper_wick  = (high_p - close_p).clip(lower=0.0)  # for shorts
        reject_bull = (close_p > open_p) & ((lower_wick / bar_range) >= wick_ratio)
        reject_bear = (close_p < open_p) & ((upper_wick / bar_range) >= wick_ratio)
    else:
        # Fallback: just directional bar
        reject_bull = close_p > open_p
        reject_bear = close_p < open_p

    # Also allow pin bar / engulfing from SMC feature set as alternative confirmation
    allow_smcpatterns = cfg.get("allow_smc_patterns", True)
    if allow_smcpatterns:
        reject_bull = reject_bull | _get_bool("pin_bar_bull") | _get_bool("engulf_bull")
        reject_bear = reject_bear | _get_bool("pin_bar_bear") | _get_bool("engulf_bear")

    # ================================================================
    # 5. RSI FILTER: Avoid already extended moves
    # ================================================================
    rsi_upper = float(cfg.get("rsi_upper", 65.0))
    rsi_lower = float(cfg.get("rsi_lower", 35.0))
    rsi       = _get_float("rsi_14", default=50.0)
    rsi_long_ok  = rsi <= rsi_upper
    rsi_short_ok = rsi >= rsi_lower

    # ================================================================
    # 6a. TREND ALIGNMENT FILTER (optional)
    #     For pullback-style entries: only buy dips when broader trend is UP.
    #     Uses a slow SMA (trend_sma_period bars) as trend reference.
    #     Long: close > slow_sma = uptrend; allow buying sd dips below fast mean
    #     Short: close < slow_sma = downtrend; allow selling sd spikes above fast mean
    # ================================================================
    trend_period = int(cfg.get("trend_sma_period", 0))  # 0 = disabled
    if trend_period > 0:
        slow_sma = close_p.rolling(trend_period, min_periods=trend_period // 2).mean()
        trend_long_ok  = close_p > slow_sma
        trend_short_ok = close_p < slow_sma
    else:
        trend_long_ok  = pd.Series(True, index=idx)
        trend_short_ok = pd.Series(True, index=idx)

    # ================================================================
    # 6b. PREMIUM/DISCOUNT STRUCTURE FILTER (optional)
    #    Longs: prefer discount zone (pd_ratio < 0.5)
    #    Shorts: prefer premium zone (pd_ratio > 0.5)
    # ================================================================
    require_pd = cfg.get("require_pd_filter", False)  # default off -- SD does equivalent job
    if require_pd:
        pd_ratio   = _get_float("pd_ratio", default=0.5)
        pd_long_ok  = pd_ratio < 0.5
        pd_short_ok = pd_ratio > 0.5
    else:
        pd_long_ok  = pd.Series(True, index=idx)
        pd_short_ok = pd.Series(True, index=idx)

    # ================================================================
    # 7. SESSION FILTER
    # ================================================================
    sess_enabled  = cfg.get("session_enabled", True)
    sess_start    = int(cfg.get("session_start_utc", 7))
    sess_end      = int(cfg.get("session_end_utc",  20))
    allowed_hours = set(cfg.get("allowed_hours_utc", list(range(sess_start, sess_end + 1))))
    if sess_enabled and len(allowed_hours) < 24:
        in_session = _session_mask(out.index, allowed_hours)
    else:
        in_session = pd.Series(True, index=idx)

    # ================================================================
    # 8. XGB REVERSAL GATE (optional)
    #    If reversal_prob columns are present (attached by pipeline after
    #    training ReversalXGBModel), apply as an additional entry filter.
    #    High XGB prob = model predicts price will reverse from SD extreme.
    # ================================================================
    xgb_cfg        = cfg.get("reversal_xgb", {})
    xgb_enabled    = xgb_cfg.get("enabled", False)
    xgb_thr_long   = float(xgb_cfg.get("threshold_long",  0.55))
    xgb_thr_short  = float(xgb_cfg.get("threshold_short", 0.55))

    has_xgb_long  = "reversal_prob_long"  in out.columns
    has_xgb_short = "reversal_prob_short" in out.columns

    if xgb_enabled and has_xgb_long:
        xgb_long_ok = out["reversal_prob_long"].fillna(0.0) >= xgb_thr_long
    else:
        xgb_long_ok = pd.Series(True, index=idx)

    if xgb_enabled and has_xgb_short:
        xgb_short_ok = out["reversal_prob_short"].fillna(0.0) >= xgb_thr_short
    else:
        xgb_short_ok = pd.Series(True, index=idx)

    # ================================================================
    # 9. COMBINE: All conditions must be true
    # ================================================================
    raw_long  = (
        sd_long_ok  & zone_long  & reject_bull &
        hmm_long_ok & rsi_long_ok  & pd_long_ok  & trend_long_ok  & in_session &
        xgb_long_ok
    )
    raw_short = (
        sd_short_ok & zone_short & reject_bear &
        hmm_short_ok & rsi_short_ok & pd_short_ok & trend_short_ok & in_session &
        xgb_short_ok
    )

    if cfg.get("long_only", False):
        raw_short = pd.Series(False, index=idx)
    if cfg.get("short_only", False):
        raw_long  = pd.Series(False, index=idx)

    # ================================================================
    # 10. COOLDOWN: Prevent signal clustering at the same zone
    # ================================================================
    cooldown_bars = int(cfg.get("cooldown_bars", 6))

    def _apply_cooldown(sig: pd.Series, n: int) -> pd.Series:
        if n <= 0:
            return sig
        sig_arr = sig.values.copy().astype(bool)
        result  = np.zeros(len(sig_arr), dtype=bool)
        last    = -n - 1
        for i in range(len(sig_arr)):
            if sig_arr[i] and (i - last) > n:
                result[i] = True
                last = i
        return pd.Series(result, index=sig.index)

    raw_long  = _apply_cooldown(raw_long,  cooldown_bars)
    raw_short = _apply_cooldown(raw_short, cooldown_bars)

    # ================================================================
    # 11. POSITION SIZING
    #     Priority:  XGB reversal prob > HMM prob > fallback pd_ratio
    #     FIXED BUG: scale from entry threshold floor, not hmm.min_confidence
    #     This ensures signals at the threshold start at base_size and
    #     only reach sizing_max at prob=1.0 — giving real differentiation.
    # ================================================================
    base_size  = float(cfg.get("base_size",  0.5))
    sizing_max = float(cfg.get("sizing_max", 1.0))

    if xgb_enabled and has_xgb_long:
        # Use XGB reversal probability for sizing (most predictive)
        # Scale from threshold floor → 1.0 to get full [base_size, sizing_max] range
        prob_floor_long  = xgb_thr_long
        prob_floor_short = xgb_thr_short
        long_conf  = out["reversal_prob_long"].fillna(prob_floor_long).clip(0.0, 1.0)
        short_conf = out["reversal_prob_short"].fillna(prob_floor_short).clip(0.0, 1.0) if has_xgb_short else pd.Series(prob_floor_short, index=idx)
    elif has_hmm:
        # FIXED: use actual min_prob entry threshold as floor (not hmm.min_confidence=0.45)
        # Previously used min_confidence=0.45 as floor, meaning ANY signal that fired
        # (at prob>=0.40) already had size >= 0.68. Now scale from the real entry floor.
        if use_inverted_hmm:
            long_conf  = out["bear_prob"].fillna(0.0).clip(0.0, 1.0)
            short_conf = out["bull_prob"].fillna(0.0).clip(0.0, 1.0)
        else:
            long_conf  = out["bull_prob"].fillna(0.0).clip(0.0, 1.0)
            short_conf = out["bear_prob"].fillna(0.0).clip(0.0, 1.0)
        prob_floor_long  = float(cfg.get("min_prob",       0.40))
        prob_floor_short = float(cfg.get("min_prob_short", 0.20))
    else:
        pd_ratio_col = _get_float("pd_ratio", 0.5)
        long_conf  = (1.0 - pd_ratio_col).clip(0.0, 1.0)
        short_conf = pd_ratio_col.clip(0.0, 1.0)
        prob_floor_long = prob_floor_short = 0.0

    prob_range_long  = max(1.0 - prob_floor_long,  1e-9)
    prob_range_short = max(1.0 - prob_floor_short, 1e-9)
    long_excess  = (long_conf  - prob_floor_long).clip(0.0,  prob_range_long)  / prob_range_long
    short_excess = (short_conf - prob_floor_short).clip(0.0, prob_range_short) / prob_range_short
    size_long    = base_size + long_excess  * (sizing_max - base_size)
    size_short   = base_size + short_excess * (sizing_max - base_size)

    # ================================================================
    # 11. EXITS: Pure SL/TP (delegated to engine)
    # ================================================================
    exit_long_sig  = pd.Series(0, index=idx)
    exit_short_sig = pd.Series(0, index=idx)

    # ================================================================
    # 12. OUTPUT
    # ================================================================
    out["signal_long"]         = raw_long.astype(int)
    out["signal_short"]        = raw_short.astype(int)
    out["exit_long"]           = exit_long_sig
    out["exit_short"]          = exit_short_sig
    out["position_size_long"]  = np.where(raw_long,  size_long,  0.0)
    out["position_size_short"] = np.where(raw_short, size_short, 0.0)
    out["signal_source"]       = np.where(
        raw_long,  "smc_sd_mean",
        np.where(raw_short, "smc_sd_mean", "")
    )

    n_long  = int(raw_long.sum())
    n_short = int(raw_short.sum())
    zone_dist = pd.Series(sd_zone.values if hasattr(sd_zone, 'values') else sd_zone, index=idx)
    n_z2_long  = int((zone_dist <= -2).sum())
    n_z3_long  = int((zone_dist <= -3).sum())
    n_z2_short = int((zone_dist >= 2).sum())
    if n_long + n_short > 0:
        print(f"  [smc_sd_mean] signals: long={n_long} short={n_short} "
              f"| zone<=-2L={n_z2_long} zone<=-3L={n_z3_long} zone>=2S={n_z2_short} "
              f"| hmm_l={int(hmm_long_ok.sum())} sd_l={int(sd_long_ok.sum())} sd_s={int(sd_short_ok.sum())}")
    else:
        print(f"  [smc_sd_mean] 0 signals "
              f"| zone<=-{min_entry_zone}L={int(sd_long_ok.sum())} zone>={min_entry_zone}S={int(sd_short_ok.sum())} "
              f"| hmm_l={int(hmm_long_ok.sum())} hmm_s={int(hmm_short_ok.sum())} "
              f"| rsi_l={int(rsi_long_ok.sum())} wick_l={int(reject_bull.sum())}")

    return out
