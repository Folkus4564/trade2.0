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
from trade2.backtesting.engine import _simulate_trades
from trade2.backtesting.costs import compute_slippage_array
from trade2.backtesting.metrics import compute_metrics


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
            sig = compute_stops(sig, atr_stop_mult, atr_tp_mult)

        if sig["signal_long"].sum() + sig["signal_short"].sum() == 0:
            return -999.0

        costs_cfg  = config["costs"]
        risk_cfg   = config["risk"]
        slippage   = compute_slippage_array(sig["Close"].astype(float), config).values
        tf_scale   = _get_tf_scale(config)
        max_hold   = risk_cfg["max_hold_bars"] * tf_scale

        equity, trades_df = _simulate_trades(
            df                   = sig,
            init_cash            = config["backtest"]["init_cash"],
            base_allocation_frac = risk_cfg["base_allocation_frac"],
            slippage             = slippage,
            commission_rt        = costs_cfg["commission_rt"],
            max_hold_bars        = max_hold,
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
