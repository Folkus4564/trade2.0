"""
Module: pipeline.py
Purpose: Main entry point - runs the full XAUUSD SMC + HMM regime strategy pipeline
Author: Strategy Code Engineer Agent
Date: 2026-03-08

Usage:
    python src/pipeline.py
    python src/pipeline.py --optimize   (runs Optuna parameter optimization)
    python src/pipeline.py --walk-forward
"""

import argparse
import json
import sys
import shutil
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader      import load_split, PROC_DIR
from src.data.features    import add_features, get_hmm_feature_matrix
from src.models.hmm_model import XAUUSDRegimeModel
from src.models.signal_generator import generate_signals, compute_stops
from src.backtesting.engine  import run_backtest, run_walk_forward
from src.backtesting.metrics import passes_criteria, verdict, format_report

STRATEGY_NAME  = "xauusd_smc_hmm_regime"
MODEL_DIR      = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
BACKTESTS_DIR  = ROOT / "backtests"
REPORTS_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# Delete stale HMM pickle so the model is retrained fresh with current features
_stale_pkl = MODEL_DIR / "hmm_regime_model.pkl"
if _stale_pkl.exists():
    _stale_pkl.unlink()
    print(f"[pipeline] Deleted stale model: {_stale_pkl}")


# ── Default hyperparameters ───────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "hma_period":            55,
    "ema_period":            21,
    "atr_period":            14,
    "rsi_period":            14,
    "adx_period":            14,
    "dc_period":             40,    # Longer channel = stronger, more reliable breakouts
    "adx_threshold":         20.0,
    "hmm_min_prob":          0.50,  # Relaxed - 2-state HMM is more decisive
    "hmm_states":            2,     # 2 states: trending (bull) vs ranging (bear)
    "atr_stop_mult":         1.5,
    "atr_tp_mult":           3.0,
    "atr_expansion_filter":  False, # Off - regime filter is sufficient gatekeeper
    "use_smc":               True,  # Enable SMC confluence signals
}


def run_pipeline(params: dict = None, walk_forward: bool = False) -> dict:
    """
    Execute the full research pipeline:
    1. Load and split data
    2. Build features (including SMC features)
    3. Train HMM on train set
    4. Generate signals on train and test sets (SMC + HMM + Donchian)
    5. Run backtests
    6. Return results dict

    Args:
        params:        Hyperparameter dict (uses DEFAULT_PARAMS if None)
        walk_forward:  Also run walk-forward validation

    Returns:
        Dict with train_metrics, test_metrics, verdict
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    print(f"\n{'='*60}")
    print(f"  XAUUSD SMC + HMM REGIME STRATEGY PIPELINE")
    print(f"{'='*60}")
    print(f"  Strategy : {STRATEGY_NAME}")
    print(f"  SMC      : {p['use_smc']}  |  DC period: {p['dc_period']}")
    print(f"  HMM prob : {p['hmm_min_prob']}  |  ATR expand: {p['atr_expansion_filter']}")
    print(f"  ATR stop : {p['atr_stop_mult']}x  |  ATR TP: {p['atr_tp_mult']}x")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[pipeline] Loading data...")
    train_raw, val_raw, test_raw = load_split("1H")
    print(f"  Train: {len(train_raw)} bars | Val: {len(val_raw)} bars | Test: {len(test_raw)} bars")

    # ── 2. Feature engineering (includes SMC features) ────────────────────────
    print("[pipeline] Engineering features (including SMC)...")
    train_feat = add_features(train_raw, hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])
    val_feat   = add_features(val_raw,   hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])
    test_feat  = add_features(test_raw,  hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])

    # Log SMC signal counts on train set
    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        if col in train_feat.columns:
            print(f"  [SMC] {col}: {int(train_feat[col].sum())} bars active (train)")

    # ── 3. Train HMM on TRAIN data only ──────────────────────────────────────
    print("[pipeline] Training HMM regime model...")
    X_train, idx_train = get_hmm_feature_matrix(train_feat)

    # Always retrain to reflect current feature set
    model = XAUUSDRegimeModel(n_states=p["hmm_states"])
    model.fit(X_train)
    model.save("hmm_regime_model")

    model.summary(X_train)

    # ── 4. Generate signals for each split ───────────────────────────────────
    def apply_signals(feat_df: pd.DataFrame, X: np.ndarray, idx: pd.Index) -> pd.DataFrame:
        labels    = model.regime_labels(X)
        bull_prob = model.bull_probability(X)
        bear_prob = model.bear_probability(X)
        sig_df    = generate_signals(
            feat_df, labels, bull_prob, bear_prob, idx,
            adx_threshold=p["adx_threshold"],
            hmm_min_prob=p["hmm_min_prob"],
            atr_expansion_filter=p["atr_expansion_filter"],
            use_smc=p["use_smc"],
        )
        sig_df = compute_stops(sig_df, p["atr_stop_mult"], p["atr_tp_mult"])
        return sig_df

    print("[pipeline] Generating signals...")
    X_val,  idx_val  = get_hmm_feature_matrix(val_feat)
    X_test, idx_test = get_hmm_feature_matrix(test_feat)

    train_sig = apply_signals(train_feat, X_train, idx_train)
    val_sig   = apply_signals(val_feat,   X_val,   idx_val)
    test_sig  = apply_signals(test_feat,  X_test,  idx_test)

    print(f"  [signals] Train: long={int(train_sig['signal_long'].sum())}, short={int(train_sig['signal_short'].sum())}")
    print(f"  [signals] Test : long={int(test_sig['signal_long'].sum())}, short={int(test_sig['signal_short'].sum())}")

    # ── 5. Backtests ──────────────────────────────────────────────────────────
    print("\n[pipeline] Running TRAIN backtest...")
    train_metrics, train_pf = run_backtest(train_sig, STRATEGY_NAME, "train")

    print("\n[pipeline] Running TEST backtest (out-of-sample)...")
    test_metrics, test_pf = run_backtest(test_sig, STRATEGY_NAME, "test")

    # ── 6. Walk-forward (optional) ────────────────────────────────────────────
    wf_results = None
    if walk_forward:
        print("\n[pipeline] Running walk-forward validation...")
        wf_results = run_walk_forward(train_sig, STRATEGY_NAME, n_windows=4)
        wf_path = BACKTESTS_DIR / f"{STRATEGY_NAME}_walkforward.json"
        with open(wf_path, "w") as f:
            json.dump(wf_results, f, indent=2)
        print(f"[pipeline] Walk-forward results: mean_return={wf_results.get('mean_return',0)*100:.1f}%")

    # ── 7. Final verdict ──────────────────────────────────────────────────────
    final_verdict = verdict(test_metrics)
    criteria      = passes_criteria(test_metrics)

    results = {
        "strategy_name": STRATEGY_NAME,
        "verdict":       final_verdict,
        "date":          str(date.today()),
        "params":        p,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "pass_criteria": criteria,
        "walk_forward":  wf_results,
        "next_iteration": _suggest_improvements(test_metrics, criteria),
    }

    # Save final verdict
    verdict_path = REPORTS_DIR / "final_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pipeline] Final verdict saved to {verdict_path}")

    # Auto-save thresholds
    ann_ret = test_metrics.get("annualized_return", 0)
    sharpe  = test_metrics.get("sharpe_ratio", 0)
    max_dd  = test_metrics.get("max_drawdown", -999)

    if ann_ret >= 0.50 and sharpe >= 1.5 and max_dd >= -0.25:
        save_name = f"{date.today()}_{STRATEGY_NAME}.py"
        save_path = ROOT / save_name
        shutil.copy(__file__, save_path)
        print(f"\n[pipeline] Strategy AUTO-SAVED (target criteria met): {save_name}")
        results["saved_as"] = save_name
    elif ann_ret >= 0.10 and sharpe >= 1.0 and max_dd >= -0.35:
        save_name = f"{date.today()}_{STRATEGY_NAME}.py"
        save_path = ROOT / save_name
        shutil.copy(__file__, save_path)
        print(f"\n[pipeline] Strategy AUTO-SAVED (minimum criteria met): {save_name}")
        results["saved_as"] = save_name

    # Summary table
    _print_summary(test_metrics, final_verdict, results.get("saved_as"))

    return results


def _suggest_improvements(metrics: dict, criteria: dict) -> list:
    """Generate improvement suggestions based on failing criteria."""
    suggestions = []
    if not criteria.get("return"):
        suggestions.append("Reduce HMM min_prob to increase SMC signal frequency")
        suggestions.append("Try HMM with 4 states to capture more granular regimes")
    if not criteria.get("sharpe"):
        suggestions.append("Reduce position size to lower volatility")
        suggestions.append("Add volatility filter: skip entries when ATR > 2x average")
    if not criteria.get("drawdown"):
        suggestions.append("Tighten ATR stop multiplier from 1.5 to 1.2")
        suggestions.append("Add max position loss circuit breaker at -5%")
    if not criteria.get("trade_count"):
        suggestions.append("Lower HMM minimum probability threshold from 0.55 to 0.45")
        suggestions.append("Widen OB validity bars from 20 to 30")
    if not criteria.get("win_rate"):
        suggestions.append("Increase ATR take profit multiplier to improve avg winner")
        suggestions.append("Add RSI confirmation filter for counter-trend exits")
    return suggestions


def _print_summary(metrics: dict, v: str, saved_as: str = None) -> None:
    """Print final results summary table."""
    SEP = "=" * 50
    print(f"\n{SEP}")
    print(f"  XAUUSD SMC+HMM PIPELINE FINAL RESULTS")
    print(f"{SEP}")
    print(f"  Annualized Return : {metrics.get('annualized_return',0)*100:>8.2f}%  (target: >=20%)")
    print(f"  Sharpe Ratio      : {metrics.get('sharpe_ratio',0):>8.3f}  (target: >=1.0)")
    print(f"  Sortino Ratio     : {metrics.get('sortino_ratio',0):>8.3f}  (target: >=2.0)")
    print(f"  Max Drawdown      : {metrics.get('max_drawdown',0)*100:>8.2f}%  (target: >=-35%)")
    print(f"  Calmar Ratio      : {metrics.get('calmar_ratio',0):>8.3f}")
    print(f"  Total Trades      : {metrics.get('total_trades','N/A'):>8}")
    print(f"  Win Rate          : {metrics.get('win_rate',0)*100:>8.2f}%")
    print(f"  Profit Factor     : {metrics.get('profit_factor',0):>8.3f}")
    if "benchmark_return" in metrics:
        print(f"  vs Buy & Hold     : {metrics.get('alpha_vs_benchmark',0)*100:>+8.2f}%")
    print(f"{SEP}")
    print(f"  Verdict: {v}")
    if saved_as:
        print(f"  Saved as: {saved_as}")
    print(f"{SEP}\n")


def optimize(n_trials: int = 100) -> dict:
    """
    Optuna hyperparameter optimization on TRAIN+VAL data only.
    Test set is never used during optimization.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[pipeline] Optuna not installed. Run: pip install optuna")
        return {}

    from src.data.loader   import load_split
    from src.data.features import add_features, get_hmm_feature_matrix
    from src.backtesting.engine import run_backtest

    train_raw, val_raw, _ = load_split("1H")

    def objective(trial):
        params = {
            "dc_period":             trial.suggest_int("dc_period", 10, 50),
            "adx_threshold":         trial.suggest_float("adx_threshold", 15.0, 35.0),
            "hmm_min_prob":          trial.suggest_float("hmm_min_prob", 0.45, 0.80),
            "atr_stop_mult":         trial.suggest_float("atr_stop_mult", 1.0, 2.5),
            "atr_tp_mult":           trial.suggest_float("atr_tp_mult", 2.0, 5.0),
            "atr_expansion_filter":  trial.suggest_categorical("atr_expansion_filter", [True, False]),
            "use_smc":               True,
        }

        try:
            feat  = add_features(val_raw, dc_period=params["dc_period"])
            X, idx = get_hmm_feature_matrix(feat)
            if len(X) < 50:
                return -1e9

            # Use already-trained model
            model = XAUUSDRegimeModel.load("hmm_regime_model")
            labels    = model.regime_labels(X)
            bull_prob = model.bull_probability(X)
            bear_prob = model.bear_probability(X)

            sig = generate_signals(feat, labels, bull_prob, bear_prob, idx,
                                   adx_threshold=params["adx_threshold"],
                                   hmm_min_prob=params["hmm_min_prob"],
                                   atr_expansion_filter=params["atr_expansion_filter"],
                                   use_smc=params["use_smc"])
            sig = compute_stops(sig, params["atr_stop_mult"], params["atr_tp_mult"])

            metrics, _ = run_backtest(sig, "optuna_trial", "val")
            sharpe = metrics.get("sharpe_ratio", -1e9)
            ret    = metrics.get("annualized_return", -1e9)
            dd     = metrics.get("max_drawdown", -999)

            if dd < -0.50:
                return -1e9

            return sharpe + ret   # Combined objective
        except Exception:
            return -1e9

    print(f"[pipeline] Optimizing {n_trials} trials on validation set...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**DEFAULT_PARAMS, **study.best_params}
    print(f"\n[pipeline] Best params: {study.best_params}")
    print(f"[pipeline] Best value: {study.best_value:.4f}")

    best_path = REPORTS_DIR / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD SMC + HMM Strategy Pipeline")
    parser.add_argument("--optimize",      action="store_true", help="Run Optuna optimization first")
    parser.add_argument("--walk-forward",  action="store_true", help="Run walk-forward validation")
    parser.add_argument("--trials",        type=int, default=100, help="Optuna trial count")
    args = parser.parse_args()

    params = None
    if args.optimize:
        params = optimize(n_trials=args.trials)

    results = run_pipeline(params=params, walk_forward=args.walk_forward)
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)
