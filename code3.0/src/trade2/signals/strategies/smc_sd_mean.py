"""
signals/strategies/smc_sd_mean.py - SMC + SD Adaptive Mean Reversion sub-strategy.

Philosophy:
  BUY when price is CHEAP (at OB/demand zone AND SD indicator shows extended below mean).
  SELL when price is EXPENSIVE (at supply/OB zone AND SD indicator shows extended above mean).

  This is a mean-reversion strategy. The SD Adaptive Mean indicator confirms that price
  has deviated significantly from its adaptive mean, providing statistical support for
  the SMC zone entry.

Entry logic (all conditions must be true):

  LONG:
    1. SD Mean: sd_smoothed <= -sd_entry_threshold  (price extended below mean)
    2. SMC zone: struct_ob_bull_retest OR dz_demand_retest  (at bullish OB or demand zone)
    3. Rejection candle: Close > Open AND lower wick >= wick_ratio * bar_range
    4. HMM inverted gate: bear_prob >= min_prob  (bear state = discount = buy zone)
    5. RSI filter: rsi_14 <= rsi_upper  (not overbought)
    6. Cooldown: no signal within last cooldown_bars bars

  SHORT (mirror):
    1. SD Mean: sd_smoothed >= +sd_entry_threshold
    2. SMC zone: struct_ob_bear_retest OR dz_supply_retest
    3. Rejection candle: Close < Open AND upper wick >= wick_ratio * bar_range
    4. HMM inverted gate: bull_prob >= min_prob_short  (bull state = premium = sell zone)
    5. RSI filter: rsi_14 >= rsi_lower
    6. Cooldown

Partial TP: handled by engine via risk.partial_tp config.
SL/TP: ATR-based, delegated to compute_stops_regime_aware() + engine.
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
    # 1. SD ADAPTIVE MEAN FILTER
    #    Enter only when price is significantly extended from mean.
    #    sd_smoothed < -threshold -> oversold / cheap -> LONG signal zone
    #    sd_smoothed > +threshold -> overbought / expensive -> SHORT signal zone
    # ================================================================
    sd_entry_threshold = float(cfg.get("sd_entry_threshold", 1.0))
    sd_smoothed = _get_float("sd_smoothed", default=0.0)

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

    use_struct_only = cfg.get("use_struct_ob_only", False)
    use_bb_zone     = cfg.get("use_bb_zone", True)  # add BB as extra zone source

    if use_struct_only:
        zone_long  = struct_ob_bull | dz_demand
        zone_short = struct_ob_bear | dz_supply
    else:
        zone_long  = struct_ob_bull | dz_demand | impulse_ob_bull
        zone_short = struct_ob_bear | dz_supply | impulse_ob_bear

    if use_bb_zone:
        zone_long  = zone_long  | bb_zone_long
        zone_short = zone_short | bb_zone_short

    # ================================================================
    # 3. HMM INVERTED REGIME GATE
    #    Reversal/mean-reversion interpretation:
    #    - High bear_prob -> discount state -> allows LONGS (cheap zone)
    #    - High bull_prob -> premium state -> allows SHORTS (expensive zone)
    # ================================================================
    min_prob_long  = float(cfg.get("min_prob",       0.55))
    min_prob_short = float(cfg.get("min_prob_short", 0.60))
    regime_gated   = cfg.get("regime_gated", False)

    has_hmm = "bear_prob" in out.columns and "bull_prob" in out.columns
    use_inverted_hmm = cfg.get("use_inverted_hmm", True)  # default: bear=buy, bull=sell
    if regime_gated and has_hmm:
        if use_inverted_hmm:
            # Mean-reversion interpretation:
            # bear_prob high = discount/oversold state = buy zone
            # bull_prob high = premium/overbought state = sell zone
            hmm_long_ok  = out["bear_prob"].fillna(0.0) >= min_prob_long
            hmm_short_ok = out["bull_prob"].fillna(0.0) >= min_prob_short
        else:
            # Momentum/trend interpretation:
            # bull_prob high = uptrend = allow longs
            # bear_prob high = downtrend = allow shorts
            hmm_long_ok  = out["bull_prob"].fillna(0.0) >= min_prob_long
            hmm_short_ok = out["bear_prob"].fillna(0.0) >= min_prob_short
    else:
        # regime_gated=false: SD mean indicator already acts as the statistical filter.
        # No HMM gate applied -- all bars pass.
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
    # 8. COMBINE: All conditions must be true
    # ================================================================
    raw_long  = (
        sd_long_ok  & zone_long  & reject_bull &
        hmm_long_ok & rsi_long_ok  & pd_long_ok  & trend_long_ok  & in_session
    )
    raw_short = (
        sd_short_ok & zone_short & reject_bear &
        hmm_short_ok & rsi_short_ok & pd_short_ok & trend_short_ok & in_session
    )

    if cfg.get("long_only", False):
        raw_short = pd.Series(False, index=idx)
    if cfg.get("short_only", False):
        raw_long  = pd.Series(False, index=idx)

    # ================================================================
    # 9. COOLDOWN: Prevent signal clustering at the same zone
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
    # 10. POSITION SIZING: HMM probability-scaled
    # ================================================================
    base_size   = float(cfg.get("base_size",   0.8))
    sizing_max  = float(cfg.get("sizing_max",  1.0))
    hmm_cfg     = config.get("hmm", {})
    min_conf    = float(hmm_cfg.get("min_confidence", 0.45))
    prob_range  = max(1.0 - min_conf, 1e-9)

    if has_hmm:
        long_conf  = out["bear_prob"].fillna(0.0).clip(0.0, 1.0)
        short_conf = out["bull_prob"].fillna(0.0).clip(0.0, 1.0)
    else:
        pd_ratio_col = _get_float("pd_ratio", 0.5)
        long_conf  = (1.0 - pd_ratio_col).clip(0.0, 1.0)
        short_conf = pd_ratio_col.clip(0.0, 1.0)

    long_excess  = (long_conf  - min_conf).clip(0.0, prob_range) / prob_range
    short_excess = (short_conf - min_conf).clip(0.0, prob_range) / prob_range
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
    if n_long + n_short > 0:
        print(f"  [smc_sd_mean] signals: long={n_long} short={n_short} "
              f"| zone_bull={int(zone_long.sum())} zone_bear={int(zone_short.sum())} "
              f"| sd_long={int(sd_long_ok.sum())} sd_short={int(sd_short_ok.sum())}")
    else:
        print(f"  [smc_sd_mean] 0 signals "
              f"| zone_bull={int(zone_long.sum())} zone_bear={int(zone_short.sum())} "
              f"| sd_long={int(sd_long_ok.sum())} sd_short={int(sd_short_ok.sum())} "
              f"| hmm_long={int(hmm_long_ok.sum())} rsi_long={int(rsi_long_ok.sum())} "
              f"| reject_bull={int(reject_bull.sum())} reject_bear={int(reject_bear.sum())}")

    return out
