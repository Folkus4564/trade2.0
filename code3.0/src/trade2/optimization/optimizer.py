"""
optimization/optimizer.py - Optuna-based hyperparameter optimization.

Optimizes "fast" signal/risk parameters against val_sharpe.
Pre-computes all slow state (data, features, regime) outside the trial loop.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from trade2.signals.generator import generate_signals, compute_stops
from trade2.backtesting.engine import _simulate_trades
from trade2.backtesting.costs import compute_slippage
from trade2.backtesting.metrics import compute_metrics


def _run_val_trial(
    val_sig_df: pd.DataFrame,    # val 5M df with regime + atr_1h already attached
    config: Dict[str, Any],
    atr_stop_mult: float,
    atr_tp_mult: float,
    hmm_min_prob: float,
    regime_persistence_bars: int,
    adx_threshold: float,
    require_pin_bar: bool,
) -> float:
    """
    Run one trial: generate signals on val, run backtest, return val_sharpe.
    Returns -999 on failure or < 10 trades.
    """
    try:
        sig = generate_signals(
            val_sig_df,
            config               = config,
            adx_threshold        = adx_threshold,
            hmm_min_prob         = hmm_min_prob,
            regime_persistence_bars = regime_persistence_bars,
            require_smc_confluence  = config["smc_5m"]["require_confluence"],
            require_pin_bar         = require_pin_bar,
        )
        sig = compute_stops(sig, atr_stop_mult, atr_tp_mult)

        if sig["signal_long"].sum() + sig["signal_short"].sum() == 0:
            return -999.0

        costs_cfg  = config["costs"]
        risk_cfg   = config["risk"]
        slippage   = compute_slippage(sig["Close"].astype(float), config)
        max_hold   = risk_cfg["max_hold_bars"] * 12  # 5M scale

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
            bars_per_year    = 252 * 24 * 12,
        )
        sharpe = metrics.get("sharpe_ratio", -999.0)
        return float(sharpe) if np.isfinite(sharpe) else -999.0

    except Exception:
        return -999.0


def run_optimization(
    val_sig_df: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: int = 100,
) -> Tuple[Dict[str, Any], float]:
    """
    Run Optuna TPE optimization over signal/risk parameters.

    Pre-conditions:
        val_sig_df must already contain regime, bull_prob, bear_prob, atr_1h columns
        (output of forward_fill_1h_regime with atr_1h).

    Returns:
        (best_params dict, best_val_sharpe)
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

    def objective(trial: "optuna.Trial") -> float:
        atr_stop_lo,  atr_stop_hi  = _bounds("atr_stop_mult",     1.0, 3.5)
        atr_tp_lo,    atr_tp_hi    = _bounds("atr_tp_mult",        3.0, 15.0)
        min_prob_lo,  min_prob_hi  = _bounds("hmm_min_prob",       0.40, 0.80)
        persist_lo,   persist_hi   = _bounds("regime_persistence", 1,   5)
        adx_lo,       adx_hi       = _bounds("adx_threshold",      15.0, 35.0)

        atr_stop_mult         = trial.suggest_float("atr_stop_mult",         atr_stop_lo,  atr_stop_hi)
        atr_tp_mult           = trial.suggest_float("atr_tp_mult",           atr_tp_lo,    atr_tp_hi)
        hmm_min_prob          = trial.suggest_float("hmm_min_prob",          min_prob_lo,  min_prob_hi)
        regime_persistence    = trial.suggest_int("regime_persistence_bars", int(persist_lo), int(persist_hi))
        adx_threshold         = trial.suggest_float("adx_threshold",         adx_lo,       adx_hi)
        require_pin_bar       = trial.suggest_categorical("require_pin_bar", [True, False])

        return _run_val_trial(
            val_sig_df           = val_sig_df,
            config               = config,
            atr_stop_mult        = atr_stop_mult,
            atr_tp_mult          = atr_tp_mult,
            hmm_min_prob         = hmm_min_prob,
            regime_persistence_bars = regime_persistence,
            adx_threshold        = adx_threshold,
            require_pin_bar      = require_pin_bar,
        )

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_params = dict(best.params)
    best_sharpe = best.value

    print(f"\n[optimizer] Best val Sharpe: {best_sharpe:.4f}")
    print(f"[optimizer] Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, best_sharpe
