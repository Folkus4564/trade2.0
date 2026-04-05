"""
optimization/optimizer.py - Optuna-based hyperparameter optimization.

Optimizes "fast" signal/risk parameters against val_sharpe.
Pre-computes all slow state (data, features, regime) outside the trial loop.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

import copy
from trade2.signals.generator import generate_signals, compute_stops, compute_stops_regime_aware
from trade2.signals.router import route_signals
from trade2.signals.regime import forward_fill_5m_to_1m
from trade2.backtesting.engine import _simulate_trades
from trade2.backtesting.costs import compute_slippage_array
from trade2.backtesting.metrics import compute_metrics
from trade2.backtesting import pullback_engine


def _get_tf_scale(config: Dict[str, Any]) -> int:
    """Return bars-per-hour for the configured signal timeframe."""
    signal_tf = config.get("strategy", {}).get("signal_timeframe", "5M")
    return {"5M": 12, "15M": 4, "30M": 2, "1H": 1, "4H": 1}.get(signal_tf, 12)


def _run_val_trial(
    val_sig_df: pd.DataFrame,    # val signal-TF df with regime + atr_1h already attached
    config: Dict[str, Any],
    atr_stop_mult: float,
    atr_tp_mult: float,
    hmm_min_prob: float,
    regime_persistence_bars: int,
    adx_threshold: float,
    require_pin_bar: bool,
    hmm_min_confidence: float = None,
    transition_cooldown_bars: int = None,
    optuna_target: str = "val_sharpe",
    # Per-regime params (regime_specialized mode only)
    trend_min_prob: float = None,
    trend_adx_threshold: float = None,
    trend_atr_stop_mult: float = None,
    range_min_prob: float = None,
    range_bb_pos_long_max: float = None,
    range_rsi_long_max: float = None,
    range_atr_stop_mult: float = None,
    range_atr_tp_mult: float = None,
) -> float:
    """
    Run one trial: generate signals on val, run backtest, return target metric.
    Returns -999 on failure or < 10 trades.
    """
    try:
        strategy_mode = config.get("strategies", {}).get("mode", "legacy")

        if strategy_mode == "regime_specialized":
            # Patch per-regime params into a config copy
            trial_config = copy.deepcopy(config)
            if trend_min_prob         is not None: trial_config["strategies"]["trend"]["min_prob"]         = trend_min_prob
            if trend_adx_threshold    is not None: trial_config["strategies"]["trend"]["adx_threshold"]    = trend_adx_threshold
            if trend_atr_stop_mult    is not None: trial_config["strategies"]["trend"]["atr_stop_mult"]    = trend_atr_stop_mult
            if range_min_prob         is not None: trial_config["strategies"]["range"]["min_prob"]         = range_min_prob
            if range_bb_pos_long_max  is not None: trial_config["strategies"]["range"]["bb_pos_long_max"]  = range_bb_pos_long_max
            if range_rsi_long_max     is not None: trial_config["strategies"]["range"]["rsi_long_max"]     = range_rsi_long_max
            if range_atr_stop_mult    is not None: trial_config["strategies"]["range"]["atr_stop_mult"]    = range_atr_stop_mult
            if range_atr_tp_mult      is not None: trial_config["strategies"]["range"]["atr_tp_mult"]      = range_atr_tp_mult
            sig = route_signals(val_sig_df, trial_config)
            sig = compute_stops_regime_aware(sig, trial_config)
        else:
            sig = generate_signals(
                val_sig_df,
                config                   = config,
                adx_threshold            = adx_threshold,
                hmm_min_prob             = hmm_min_prob,
                regime_persistence_bars  = regime_persistence_bars,
                require_smc_confluence   = config["smc_5m"]["require_confluence"],
                require_pin_bar          = require_pin_bar,
                hmm_min_confidence       = hmm_min_confidence,
                transition_cooldown_bars = transition_cooldown_bars,
            )
            sig = compute_stops(sig, atr_stop_mult, atr_tp_mult,
                                use_signal_atr=config["risk"]["use_signal_atr"])

        if sig["signal_long"].sum() + sig["signal_short"].sum() == 0:
            return -999.0

        costs_cfg  = config["costs"]
        risk_cfg   = config["risk"]
        slippage   = compute_slippage_array(sig["Close"].astype(float), config).values
        tf_scale   = _get_tf_scale(config)
        max_hold   = risk_cfg["max_hold_bars"] * tf_scale
        be_trigger = risk_cfg.get("break_even_atr_trigger", 0.0)

        equity, trades_df = _simulate_trades(
            df                   = sig,
            init_cash            = config["backtest"]["init_cash"],
            base_allocation_frac = risk_cfg["base_allocation_frac"],
            slippage             = slippage,
            commission_rt        = costs_cfg["commission_rt"],
            max_hold_bars        = max_hold,
            be_atr_trigger       = be_trigger,
        )

        if len(trades_df) < 10:
            return -999.0

        bh_equity = config["backtest"]["init_cash"] * (sig["Close"] / sig["Close"].iloc[0])
        metrics = compute_metrics(
            equity_curve     = equity,
            trades           = trades_df[["pnl", "duration_bars"]],
            benchmark_equity = bh_equity,
            bars_per_year    = 252 * 24 * tf_scale,
        )
        if optuna_target == "val_return":
            val = metrics.get("annualized_return", -999.0)
        else:
            val = metrics.get("sharpe_ratio", -999.0)
        return float(val) if np.isfinite(val) else -999.0

    except Exception:
        return -999.0


def run_optimization(
    val_sig_df: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 100,
    train_sig_df: pd.DataFrame = None,
    optuna_target: str = "val_sharpe",
) -> Tuple[Dict[str, Any], float]:
    """
    Run Optuna TPE optimization over signal/risk parameters.

    Objective: minimize overfitting by using combined train+val Sharpe.
    If train_sig_df provided: objective = min(train_sharpe, val_sharpe).
    If only val_sig_df: objective = val_sharpe.

    Pre-conditions:
        sig_dfs must already contain regime, bull_prob, bear_prob, atr_1h columns.

    Returns:
        (best_params dict, best_objective_score)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna is required for optimization. Run: pip install optuna")

    opt_cfg    = config.get("optimization", {})
    search     = opt_cfg.get("search_space", {})

    def _bounds(key, default_lo, default_hi):
        v = search.get(key, [default_lo, default_hi])
        return v[0], v[1]

    strategy_mode = config.get("strategies", {}).get("mode", "legacy")

    def objective(trial: "optuna.Trial") -> float:
        atr_stop_lo,  atr_stop_hi  = _bounds("atr_stop_mult",           1.0, 3.5)
        atr_tp_lo,    atr_tp_hi    = _bounds("atr_tp_mult",             3.0, 15.0)
        min_prob_lo,  min_prob_hi  = _bounds("hmm_min_prob",            0.40, 0.80)
        persist_lo,   persist_hi   = _bounds("regime_persistence",      1,   5)
        adx_lo,       adx_hi       = _bounds("adx_threshold",           15.0, 35.0)
        conf_lo,      conf_hi      = _bounds("hmm_min_confidence",      0.35, 0.65)
        cool_lo,      cool_hi      = _bounds("transition_cooldown_bars", 0,   4)

        atr_stop_mult            = trial.suggest_float("atr_stop_mult",          atr_stop_lo, atr_stop_hi)
        atr_tp_mult              = trial.suggest_float("atr_tp_mult",            atr_tp_lo,   atr_tp_hi)
        hmm_min_prob             = trial.suggest_float("hmm_min_prob",           min_prob_lo, min_prob_hi)
        regime_persistence       = trial.suggest_int("regime_persistence_bars",  int(persist_lo), int(persist_hi))
        adx_threshold            = trial.suggest_float("adx_threshold",          adx_lo,      adx_hi)
        require_pin_bar          = trial.suggest_categorical("require_pin_bar",  [True, False])
        hmm_min_confidence       = trial.suggest_float("hmm_min_confidence",     conf_lo,     conf_hi)
        transition_cooldown      = trial.suggest_int("transition_cooldown_bars", int(cool_lo), int(cool_hi))

        trial_kwargs = dict(
            config                   = config,
            atr_stop_mult            = atr_stop_mult,
            atr_tp_mult              = atr_tp_mult,
            hmm_min_prob             = hmm_min_prob,
            regime_persistence_bars  = regime_persistence,
            adx_threshold            = adx_threshold,
            require_pin_bar          = require_pin_bar,
            hmm_min_confidence       = hmm_min_confidence,
            transition_cooldown_bars = transition_cooldown,
        )

        # Per-regime params (only when regime_specialized mode is active)
        if strategy_mode == "regime_specialized":
            t_min_prob_lo, t_min_prob_hi = _bounds("trend_min_prob",         0.55, 0.90)
            t_adx_lo,      t_adx_hi      = _bounds("trend_adx_threshold",   15.0, 35.0)
            t_stop_lo,     t_stop_hi     = _bounds("trend_atr_stop_mult",    1.5,  4.0)
            r_min_prob_lo, r_min_prob_hi = _bounds("range_min_prob",         0.40, 0.80)
            r_bb_lo,       r_bb_hi       = _bounds("range_bb_pos_long_max",  0.05, 0.30)
            r_rsi_lo,      r_rsi_hi      = _bounds("range_rsi_long_max",     25.0, 45.0)
            r_stop_lo,     r_stop_hi     = _bounds("range_atr_stop_mult",    0.75, 2.5)
            r_tp_lo,       r_tp_hi       = _bounds("range_atr_tp_mult",      1.5,  5.0)

            trial_kwargs["trend_min_prob"]        = trial.suggest_float("trend_min_prob",        t_min_prob_lo, t_min_prob_hi)
            trial_kwargs["trend_adx_threshold"]   = trial.suggest_float("trend_adx_threshold",   t_adx_lo,     t_adx_hi)
            trial_kwargs["trend_atr_stop_mult"]   = trial.suggest_float("trend_atr_stop_mult",   t_stop_lo,    t_stop_hi)
            trial_kwargs["range_min_prob"]        = trial.suggest_float("range_min_prob",        r_min_prob_lo, r_min_prob_hi)
            trial_kwargs["range_bb_pos_long_max"] = trial.suggest_float("range_bb_pos_long_max", r_bb_lo,      r_bb_hi)
            trial_kwargs["range_rsi_long_max"]    = trial.suggest_float("range_rsi_long_max",    r_rsi_lo,     r_rsi_hi)
            trial_kwargs["range_atr_stop_mult"]   = trial.suggest_float("range_atr_stop_mult",   r_stop_lo,    r_stop_hi)
            trial_kwargs["range_atr_tp_mult"]     = trial.suggest_float("range_atr_tp_mult",     r_tp_lo,      r_tp_hi)

        # ---- Scalp-specific params: patch config when present in search_space ----
        _SCALP_PARAMS = ("session_start_hour", "session_end_hour",
                         "persistence_bars_short", "break_even_atr_trigger")
        if any(k in search for k in _SCALP_PARAMS):
            trial_config = copy.deepcopy(config)
            if "session_start_hour" in search or "session_end_hour" in search:
                ss_lo, ss_hi = _bounds("session_start_hour", 6, 10)
                se_lo, se_hi = _bounds("session_end_hour", 16, 22)
                sess_start = trial.suggest_int("session_start_hour", int(ss_lo), int(ss_hi))
                sess_end   = trial.suggest_int("session_end_hour",   int(se_lo), int(se_hi))
                trial_config["session"]["allowed_hours_utc"] = list(range(sess_start, sess_end + 1))
            if "persistence_bars_short" in search:
                pb_lo, pb_hi = _bounds("persistence_bars_short", 1, 6)
                pb_short = trial.suggest_int("persistence_bars_short", int(pb_lo), int(pb_hi))
                trial_config["regime"]["persistence_bars_short"] = pb_short
            if "break_even_atr_trigger" in search:
                be_lo, be_hi = _bounds("break_even_atr_trigger", 0.0, 1.5)
                be_val = trial.suggest_float("break_even_atr_trigger", be_lo, be_hi)
                trial_config["risk"]["break_even_atr_trigger"] = be_val
            trial_kwargs["config"] = trial_config

        trial_kwargs["optuna_target"] = optuna_target
        val_score = _run_val_trial(val_sig_df=val_sig_df, **trial_kwargs)
        if train_sig_df is not None and val_score > -990:
            train_score = _run_val_trial(val_sig_df=train_sig_df, **trial_kwargs)
            if train_score <= -990:
                return val_score * 0.5  # penalize if train fails
            # Combined: take the minimum (most conservative)
            return min(val_score, train_score)
        return val_score

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_params = dict(best.params)
    best_sharpe = best.value

    print(f"\n[optimizer] Best {optuna_target}: {best_sharpe:.4f}")
    print(f"[optimizer] Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, best_sharpe


def run_optimization_smc_pr(
    val_5m_raw: pd.DataFrame,
    train_5m_raw: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 200,
    optuna_target: str = "val_sharpe",
) -> Dict[str, Any]:
    """
    Two-tier optimization for the SMC Pullback Reversal strategy.

    Outer grid: swing_structure_size values [20, 30, 40, 50, 60] -- requires feature
    recomputation because the swing structure detection window changes the feature columns.

    Inner Optuna: fast signal/risk params that can be varied without re-running features:
    - atr_stop_mult, atr_tp_mult, entry_cooldown_bars, trailing params
    - require_premium_discount, require_internal_choch, first_retest_only, zone_edge_prox
    - min_wick_body_ratio, min_engulf_body_atr

    Args:
        val_5m_raw:   Raw (no-signal) val 5M DataFrame with regime columns attached.
        train_5m_raw: Same for train split (for anti-overfit scoring).
        config:       Full merged config dict.
        n_trials:     Total Optuna trials (split evenly across swing sizes).
        optuna_target: 'val_sharpe' | 'val_return'.

    Returns:
        dict with best_params, best_value, best_swing_size.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna required: pip install optuna")

    from trade2.features.builder import add_5m_features
    from trade2.signals.router import route_signals
    from trade2.signals.generator import compute_stops_regime_aware
    from trade2.backtesting.engine import _simulate_trades
    from trade2.backtesting.costs import compute_slippage_array
    from trade2.backtesting.metrics import compute_metrics

    opt_cfg = config.get("optimization", {})
    ss      = opt_cfg.get("search_space", {})

    swing_sizes = [20, 30, 40, 50, 60]
    trials_per  = max(10, n_trials // len(swing_sizes))
    best_overall = {"best_value": -999.0, "best_params": {}, "best_swing_size": 50}

    def _run_split(feat_df: pd.DataFrame, trial_cfg: Dict[str, Any]) -> float:
        try:
            sig = route_signals(feat_df, trial_cfg)
            sig = compute_stops_regime_aware(sig, trial_cfg)
            if sig["signal_long"].sum() + sig["signal_short"].sum() < 10:
                return -999.0
            risk_cfg   = trial_cfg["risk"]
            costs_cfg  = trial_cfg["costs"]
            slippage   = compute_slippage_array(sig["Close"].astype(float), trial_cfg).values
            tf_scale   = _get_tf_scale(trial_cfg)
            equity, trades_df = _simulate_trades(
                df                   = sig,
                init_cash            = trial_cfg["backtest"]["init_cash"],
                base_allocation_frac = risk_cfg["base_allocation_frac"],
                slippage             = slippage,
                commission_rt        = costs_cfg["commission_rt"],
                max_hold_bars        = risk_cfg["max_hold_bars"] * tf_scale,
                be_atr_trigger       = risk_cfg.get("break_even_atr_trigger", 0.0),
            )
            if len(trades_df) < 10:
                return -999.0
            bh = trial_cfg["backtest"]["init_cash"] * (sig["Close"] / sig["Close"].iloc[0])
            metrics = compute_metrics(equity, trades_df[["pnl", "duration_bars"]], bh,
                                      bars_per_year=252 * 12)
            v = metrics.get("sharpe_ratio" if optuna_target == "val_sharpe" else "annualized_return", -999.0)
            return float(v) if np.isfinite(v) else -999.0
        except Exception:
            return -999.0

    for swing_size in swing_sizes:
        # Build features for this swing size
        swing_cfg = copy.deepcopy(config)
        swing_cfg["smc_luxalgo_5m"]["swing_structure_size"] = swing_size

        print(f"[optuna-smc-pr] swing_size={swing_size} -- building features ...")
        val_feat   = add_5m_features(val_5m_raw,   swing_cfg)
        train_feat = add_5m_features(train_5m_raw, swing_cfg) if train_5m_raw is not None else None

        def objective(trial: "optuna.Trial") -> float:
            trial_cfg = copy.deepcopy(swing_cfg)
            smc_pr    = trial_cfg["strategies"]["smc_pullback_reversal"]
            lux_5m    = trial_cfg["smc_luxalgo_5m"]

            def _fb(key, lo, hi):
                r = ss.get(key, [lo, hi])
                return trial.suggest_float(key, r[0], r[1])
            def _ib(key, lo, hi):
                r = ss.get(key, [lo, hi])
                return trial.suggest_int(key, int(r[0]), int(r[1]))
            def _cb(key, choices):
                r = ss.get(key, choices)
                return trial.suggest_categorical(key, r if isinstance(r, list) else choices)

            smc_pr["atr_stop_mult"]           = _fb("smc_pr_atr_stop_mult",    0.8,  2.5)
            smc_pr["atr_tp_mult"]             = _fb("smc_pr_atr_tp_mult",      1.5,  5.0)
            smc_pr["entry_cooldown_bars"]     = _ib("smc_pr_entry_cooldown",   0,    5)
            smc_pr["trailing_enabled"]        = _cb("smc_pr_trailing_enabled", [True, False])
            smc_pr["trailing_atr_mult"]       = _fb("smc_pr_trailing_atr_mult",0.5,  1.5)
            smc_pr["require_premium_discount"]= _cb("smc_pr_require_premium_discount", [True, False])
            smc_pr["require_internal_choch"]  = _cb("smc_pr_require_internal_choch",   [True, False])
            smc_pr["internal_choch_lookback"] = _ib("smc_pr_internal_choch_lookback",  3, 10)
            smc_pr["first_retest_only"]       = _cb("smc_pr_first_retest_only",        [True, False])
            smc_pr["zone_edge_atr_proximity"] = _fb("smc_pr_zone_edge_prox",    0.0,  0.5)
            smc_pr["min_wick_body_ratio"]     = _fb("smc_pr_min_wick_ratio",    2.0,  3.5)
            smc_pr["min_engulf_body_atr"]     = _fb("smc_pr_min_engulf_body",   0.0,  0.5)
            lux_5m["confluence_filter"]       = _cb("smc_pr_confluence_filter", [True, False])

            val_score = _run_split(val_feat, trial_cfg)
            if train_feat is not None and val_score > -990:
                train_score = _run_split(train_feat, trial_cfg)
                if train_score <= -990:
                    return val_score * 0.5
                return min(val_score, train_score)
            return val_score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=trials_per, show_progress_bar=False)

        print(f"[optuna-smc-pr] swing={swing_size} best {optuna_target}={study.best_value:.4f}")

        if study.best_value > best_overall["best_value"]:
            best_overall["best_value"]     = study.best_value
            best_overall["best_params"]    = dict(study.best_params)
            best_overall["best_swing_size"] = swing_size

    print(f"\n[optuna-smc-pr] Overall best {optuna_target}: {best_overall['best_value']:.4f}")
    print(f"[optuna-smc-pr] Best swing_size: {best_overall['best_swing_size']}")
    print(f"[optuna-smc-pr] Best params: {best_overall['best_params']}")

    return best_overall


def run_optimization_3tf(
    val_5m_signals: pd.DataFrame,
    val_1m: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 200,
    optuna_target: str = "val_sharpe",
) -> Dict[str, Any]:
    """
    Optuna optimization for the 3-TF pullback strategy.

    Pre-computes forward-filled 1M base once. Each trial varies pullback + risk
    params, re-runs pullback_engine.simulate(), returns target metric.

    Args:
        val_5m_signals: val 5M df with signals + stops already computed
        val_1m:         val 1M OHLCV (raw, not yet forward-filled)
        config:         full config dict
        n_trials:       number of Optuna trials
        optuna_target:  'val_sharpe', 'val_return', or 'val_calmar'

    Returns:
        dict with keys: best_params, best_value, study
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    costs_cfg = config["costs"]
    risk_cfg  = config["risk"]
    ss        = config.get("optimization", {}).get("search_space", {})

    # Pre-compute forward-fill once -- doesn't change between trials
    val_1m_base = forward_fill_5m_to_1m(val_1m, val_5m_signals)

    # Pre-compute slippage once
    slippage_arr = compute_slippage_array(val_1m_base["Close"], config)
    commission_rt = costs_cfg["commission_rt_bps"] / 10000.0 * 2
    init_cash     = risk_cfg["init_cash"]

    def objective(trial: "optuna.Trial") -> float:
        pb_trial   = copy.deepcopy(config["pullback"])
        risk_trial = copy.deepcopy(risk_cfg)

        # Sample pullback params from search space
        _float_pb_keys = [
            "bull_tier1_mult", "bull_tier2_mult",
            "bear_tier1_mult", "bear_tier2_mult",
            "neutral_tier1_mult", "neutral_tier2_mult",
            "runaway_atr_mult",
        ]
        for key in _float_pb_keys:
            if key in ss:
                pb_trial[key] = trial.suggest_float(key, ss[key][0], ss[key][1])

        if "max_wait_bars" in ss:
            pb_trial["max_wait_bars"] = trial.suggest_int(
                "max_wait_bars", int(ss["max_wait_bars"][0]), int(ss["max_wait_bars"][1])
            )

        # Enforce tier2 < tier1 per regime
        for prefix in ("bull_", "bear_", "neutral_"):
            t1_key = f"{prefix}tier1_mult"
            t2_key = f"{prefix}tier2_mult"
            if pb_trial[t2_key] >= pb_trial[t1_key]:
                pb_trial[t2_key] = pb_trial[t1_key] * 0.5

        # Sample risk params
        if "atr_stop_mult" in ss:
            risk_trial["atr_stop_mult"] = trial.suggest_float(
                "atr_stop_mult", ss["atr_stop_mult"][0], ss["atr_stop_mult"][1]
            )
        if "atr_tp_mult" in ss:
            risk_trial["atr_tp_mult"] = trial.suggest_float(
                "atr_tp_mult", ss["atr_tp_mult"][0], ss["atr_tp_mult"][1]
            )
        if "trailing_atr_mult" in ss:
            risk_trial["trailing_atr_mult"] = trial.suggest_float(
                "trailing_atr_mult", ss["trailing_atr_mult"][0], ss["trailing_atr_mult"][1]
            )

        try:
            equity, trades_df = pullback_engine.simulate(
                val_1m_base, pb_trial, risk_trial, init_cash, commission_rt, slippage_arr,
            )
            if len(trades_df) < 10:
                return -999.0
            metrics = compute_metrics(equity, trades_df, bars_per_year=98280)
            target_map = {
                "val_sharpe": metrics.get("sharpe_ratio",       -999.0),
                "val_return": metrics.get("annualized_return",  -999.0),
                "val_calmar": metrics.get("calmar_ratio",       -999.0),
            }
            return float(target_map.get(optuna_target, metrics.get("sharpe_ratio", -999.0)))
        except Exception:
            return -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[optuna-3tf] Best {optuna_target}: {study.best_value:.4f}")
    print(f"[optuna-3tf] Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }


def run_optimization_smc_sd_mean(
    val_feat_df: pd.DataFrame,
    train_feat_df: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 200,
    optuna_target: str = "val_sharpe",
) -> Dict[str, Any]:
    """
    Optuna optimization for the SMC + SD Adaptive Mean strategy.

    SD Adaptive Mean parameters (sd_length_short, sd_length_long, sd_entry_threshold, etc.)
    require feature recomputation because they change rolling window sizes. This function
    re-computes sd_adaptive_mean features inside each trial (cheap -- just rolling ops).

    Other parameters (atr_stop_mult, atr_tp_mult, min_prob, cooldown, wick_ratio) are
    applied directly to the signal generation config without feature recomputation.

    Args:
        val_feat_df:   val 5M DataFrame with all features EXCEPT sd_adaptive_mean columns.
                       (sd_smoothed, sd_band_*, sd_zone will be recomputed per trial)
        train_feat_df: train split (same format), used for anti-overfitting scoring.
        config:        Full merged config dict.
        n_trials:      Number of Optuna trials.
        optuna_target: 'val_sharpe' | 'val_return'.

    Returns:
        dict with best_params, best_value, study.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna required: pip install optuna")

    from trade2.features.sd_adaptive_mean import compute_sd_adaptive_mean
    from trade2.features.hmm_features import add_hmm_features
    from trade2.signals.router import route_signals
    from trade2.signals.generator import compute_stops_regime_aware
    from trade2.backtesting.engine import run_backtest
    from trade2.backtesting.costs import compute_slippage_array
    from trade2.backtesting.metrics import compute_metrics

    opt_cfg = config.get("optimization", {})
    ss      = opt_cfg.get("search_space", {})

    def _b(key, lo, hi, is_int=False):
        r = ss.get(key, [lo, hi])
        if is_int:
            return trial.suggest_int(key, int(r[0]), int(r[1]))
        return trial.suggest_float(key, r[0], r[1])

    def _run_split(feat_df: pd.DataFrame, trial_cfg: Dict[str, Any]) -> float:
        try:
            # Re-compute SD Adaptive Mean with trial params
            df_sd = compute_sd_adaptive_mean(feat_df, trial_cfg)
            # Re-compute HMM feature columns that depend on sd_smoothed
            # (hmm_feat_sd_distance = sd_smoothed since both already lagged)
            if "sd_smoothed" in df_sd.columns:
                df_sd["hmm_feat_sd_distance"] = df_sd["sd_smoothed"]
            # Generate signals
            sig = route_signals(df_sd, trial_cfg)
            sig = compute_stops_regime_aware(sig, trial_cfg)
            n_signals = sig["signal_long"].sum() + sig["signal_short"].sum()
            if n_signals < 10:
                return -999.0
            # Run backtest (uses partial TP engine from config)
            risk_cfg  = trial_cfg["risk"]
            costs_cfg = trial_cfg["costs"]
            slippage  = compute_slippage_array(sig["Close"].astype(float), trial_cfg).values
            tf_scale  = 12  # 5M = 12 bars per hour
            bars_per_year = 252 * 24 * tf_scale

            from trade2.backtesting.engine import _simulate_trades_partial_tp, _simulate_trades_multi, _simulate_trades

            max_concurrent = int(risk_cfg.get("max_concurrent_positions", 1))
            partial_tp_cfg = risk_cfg.get("partial_tp", None)
            use_partial_tp = partial_tp_cfg is not None and partial_tp_cfg.get("levels", 1) > 1

            init_cash  = trial_cfg["backtest"]["init_cash"]
            base_alloc = risk_cfg["base_allocation_frac"]
            comm       = costs_cfg["commission_rt"]
            max_hold   = risk_cfg["max_hold_bars"] * tf_scale
            be_trigger = risk_cfg.get("break_even_atr_trigger", 0.0)
            contract_oz = trial_cfg["backtest"]["contract_size_oz"]

            if use_partial_tp:
                tp_factors  = partial_tp_cfg["tp_factors"]
                size_fracs  = partial_tp_cfg["size_fracs"]
                be_after_tp1 = partial_tp_cfg.get("be_after_tp1", True)
                n_levels    = len(tp_factors)
                eff_conc    = max_concurrent * n_levels
                equity, trades_df = _simulate_trades_partial_tp(
                    sig, init_cash, base_alloc, slippage, comm, max_hold, be_trigger,
                    contract_oz, eff_conc, tp_factors, size_fracs, be_after_tp1,
                )
            elif max_concurrent > 1:
                equity, trades_df = _simulate_trades_multi(
                    sig, init_cash, base_alloc, slippage, comm, max_hold, be_trigger,
                    contract_oz, max_concurrent,
                )
            else:
                equity, trades_df = _simulate_trades(
                    sig, init_cash, base_alloc, slippage, comm, max_hold, be_trigger,
                    contract_oz,
                )

            if len(trades_df) < 10:
                return -999.0

            bh = init_cash * (sig["Close"] / sig["Close"].iloc[0])
            metrics = compute_metrics(equity, trades_df[["pnl", "duration_bars"]], bh,
                                      bars_per_year=bars_per_year)
            if optuna_target == "val_return":
                v = metrics.get("annualized_return", -999.0)
            else:
                v = metrics.get("sharpe_ratio", -999.0)
            return float(v) if np.isfinite(v) else -999.0
        except Exception as _e:
            import traceback as _tb
            if not hasattr(_run_split, "_logged"):
                _run_split._logged = True
                print(f"[optuna-smc-sd-mean] _run_split error (first occurrence): {_e}")
                _tb.print_exc()
            return -999.0

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = copy.deepcopy(config)
        smc_sd = trial_cfg["strategies"]["smc_sd_mean"]
        risk   = trial_cfg["risk"]

        # SD Adaptive Mean params (require feature recomputation)
        smc_sd["sd_length_short"]     = trial.suggest_int("sd_length_short",
                                            *[int(v) for v in ss.get("sd_length_short", [8, 30])])
        smc_sd["sd_length_long"]      = trial.suggest_int("sd_length_long",
                                            *[int(v) for v in ss.get("sd_length_long", [30, 100])])
        # Ensure short < long
        if smc_sd["sd_length_short"] >= smc_sd["sd_length_long"]:
            smc_sd["sd_length_long"] = smc_sd["sd_length_short"] + 10
        smc_sd["sd_atr_threshold"]    = trial.suggest_float("sd_atr_threshold",
                                            *ss.get("sd_atr_threshold", [0.5, 3.0]))
        smc_sd["sd_smooth_length"]    = trial.suggest_int("sd_smooth_length",
                                            *[int(v) for v in ss.get("sd_smooth_length", [10, 40])])
        smc_sd["sd_smooth_sd_length"] = trial.suggest_int("sd_smooth_sd_length",
                                            *[int(v) for v in ss.get("sd_smooth_sd_length", [20, 80])])
        smc_sd["sd_entry_threshold"]  = trial.suggest_float("sd_entry_threshold",
                                            *ss.get("sd_entry_threshold", [0.5, 2.5]))

        # Signal params (no feature recomputation)
        smc_sd["atr_stop_mult"] = trial.suggest_float("atr_stop_mult",
                                      *ss.get("atr_stop_mult", [0.8, 2.0]))
        smc_sd["atr_tp_mult"]   = trial.suggest_float("atr_tp_mult",
                                      *ss.get("atr_tp_mult", [0.8, 2.0]))
        smc_sd["min_prob"]      = trial.suggest_float("min_prob",
                                      *ss.get("hmm_min_prob", [0.45, 0.80]))
        smc_sd["min_prob_short"] = smc_sd["min_prob"]
        smc_sd["wick_ratio"]    = trial.suggest_float("wick_ratio",
                                      *ss.get("wick_ratio", [0.1, 0.5]))
        smc_sd["cooldown_bars"] = trial.suggest_int("cooldown_bars",
                                      *[int(v) for v in ss.get("cooldown_bars", [3, 15])])
        # Optional: bb_zone_threshold (if use_bb_zone=True)
        if smc_sd.get("use_bb_zone", False) and "bb_zone_threshold" in ss:
            smc_sd["bb_zone_threshold"] = trial.suggest_float("bb_zone_threshold",
                                              *ss["bb_zone_threshold"])
        # Optional: rsi_upper filter
        if "rsi_upper" in ss:
            smc_sd["rsi_upper"] = trial.suggest_float("rsi_upper", *ss["rsi_upper"])

        val_score = _run_split(val_feat_df, trial_cfg)
        if train_feat_df is not None and val_score > -990:
            train_score = _run_split(train_feat_df, trial_cfg)
            if train_score <= -990:
                return val_score * 0.5
            # Soft penalty: allow moderate train losses but reward val improvement.
            # val_score + 0.2 * min(0, train_score):
            #   train=-6, val=+1.0 -> 1.0 - 1.2 = -0.2 (just below zero, optimizer keeps searching)
            #   train=-2, val=+0.5 -> 0.5 - 0.4 = +0.1 (positive! optimizer converges here)
            #   train=+0.3, val=+0.8 -> 0.8 + 0 = +0.8 (best case)
            return val_score + min(0.0, train_score) * 0.2
        return val_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n[optuna-smc-sd-mean] Best {optuna_target}: {study.best_value:.4f}")
    print(f"[optuna-smc-sd-mean] Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
    }
