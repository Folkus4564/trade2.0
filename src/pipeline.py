"""
Module: pipeline.py
Purpose: Main entry point - runs the full XAUUSD SMC + HMM regime strategy pipeline
Author: Strategy Code Engineer Agent
Date: 2026-03-10 (v2 - config-driven, experiment logging, hard rejection, auto-export)

Usage:
    python src/pipeline.py
    python src/pipeline.py --optimize   (runs Optuna parameter optimization)
    python src/pipeline.py --walk-forward
    python src/pipeline.py --skip-walk-forward  (walk-forward is now DEFAULT)
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.config           import load_config, get_config
from src.data.loader      import load_split, PROC_DIR, dataset_version, audit_missing_bars, _raw_csv_path
from src.data.features    import add_features, get_hmm_feature_matrix
from src.models.hmm_model import XAUUSDRegimeModel
from src.models.signal_generator import generate_signals, compute_stops
from src.backtesting.engine  import run_backtest, run_walk_forward, run_backtest_2x_costs
from src.backtesting.metrics import (
    passes_criteria, verdict, format_report,
    multi_split_verdict, hard_rejection_checks,
)
from src.experiment import ExperimentLogger
from src.export     import export_approved_strategy

# ── Boot config ───────────────────────────────────────────────────────────────
CFG = load_config()

STRATEGY_NAME  = "xauusd_smc_hmm_regime"
MODEL_DIR      = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
BACKTESTS_DIR  = ROOT / "backtests"
REPORTS_DIR.mkdir(exist_ok=True)

np.random.seed(CFG.get("hmm", {}).get("random_seed", 42))

# Delete stale HMM pickle so the model is retrained fresh with current features
_stale_pkl = MODEL_DIR / "hmm_regime_model.pkl"
if _stale_pkl.exists():
    _stale_pkl.unlink()
    print(f"[pipeline] Deleted stale model: {_stale_pkl}")


def _build_params_from_config() -> dict:
    """Build default hyperparameter dict from config sections."""
    feat = CFG.get("features", {})
    smc  = CFG.get("smc",  {})
    risk = CFG.get("risk", {})
    hmm  = CFG.get("hmm",  {})
    reg  = CFG.get("regime", {})
    return {
        "hma_period":            feat.get("hma_period",  55),
        "ema_period":            feat.get("ema_period",  21),
        "atr_period":            feat.get("atr_period",  14),
        "rsi_period":            feat.get("rsi_period",  14),
        "adx_period":            feat.get("adx_period",  14),
        "dc_period":             feat.get("dc_period",   40),
        "adx_threshold":         reg.get("adx_threshold", 20.0),
        "hmm_min_prob":          hmm.get("min_prob_hard",  0.50),
        "hmm_states":            hmm.get("n_states",       3),
        "atr_stop_mult":         risk.get("atr_stop_mult", 1.5),
        "atr_tp_mult":           risk.get("atr_tp_mult",   3.0),
        "atr_expansion_filter":  False,
        "use_smc":               True,
    }


def _regime_distribution(labels: pd.Series) -> dict:
    """Compute regime label distribution as fractions."""
    counts = labels.value_counts(normalize=True)
    return {regime: round(float(frac), 4) for regime, frac in counts.items()}


def run_pipeline(
    params: dict = None,
    walk_forward: bool = True,      # DEFAULT is True now (Phase B4)
) -> dict:
    """
    Execute the full research pipeline:
    1. Load and split data (with gap audit)
    2. Build features
    3. Train HMM on train set
    4. Generate signals on all splits
    5. Run backtests + 2x cost sensitivity
    6. Walk-forward (default ON)
    7. Hard rejection checks
    8. Experiment logging
    9. Auto-export if APPROVED
    """
    p = {**_build_params_from_config(), **(params or {})}

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

    # Gap audit (Phase F3)
    raw_path = _raw_csv_path()
    gap_audit = audit_missing_bars(train_raw, "1h")
    print(f"  [data] Missing bars: {gap_audit['missing_bars']} ({gap_audit['missing_pct']}%)")

    # Dataset version / SHA256 (Phase F2)
    data_ver = dataset_version(raw_path)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("[pipeline] Engineering features (including SMC)...")
    train_feat = add_features(train_raw, hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])
    val_feat   = add_features(val_raw,   hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])
    test_feat  = add_features(test_raw,  hma_period=p["hma_period"], ema_period=p["ema_period"],
                               dc_period=p["dc_period"], atr_period=p["atr_period"])

    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        if col in train_feat.columns:
            print(f"  [SMC] {col}: {int(train_feat[col].sum())} bars active (train)")

    # ── 3. Train HMM on TRAIN data only ──────────────────────────────────────
    print("[pipeline] Training HMM regime model...")
    X_train, idx_train = get_hmm_feature_matrix(train_feat)

    model = XAUUSDRegimeModel(n_states=p["hmm_states"])
    model.fit(X_train)
    model.save("hmm_regime_model")
    model.summary(X_train)

    # ── 4. Generate signals for each split ───────────────────────────────────
    def apply_signals(feat_df, X, idx):
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
        return sig_df, labels

    print("[pipeline] Generating signals...")
    X_val,  idx_val  = get_hmm_feature_matrix(val_feat)
    X_test, idx_test = get_hmm_feature_matrix(test_feat)

    train_sig, train_labels = apply_signals(train_feat, X_train, idx_train)
    val_sig,   val_labels   = apply_signals(val_feat,   X_val,   idx_val)
    test_sig,  test_labels  = apply_signals(test_feat,  X_test,  idx_test)

    # Regime distributions (for drift check)
    train_regime_dist = _regime_distribution(train_labels)
    test_regime_dist  = _regime_distribution(test_labels)
    print(f"  [regime] Train dist: {train_regime_dist}")
    print(f"  [regime] Test  dist: {test_regime_dist}")

    print(f"  [signals] Train: long={int(train_sig['signal_long'].sum())}, short={int(train_sig['signal_short'].sum())}")
    print(f"  [signals] Test : long={int(test_sig['signal_long'].sum())}, short={int(test_sig['signal_short'].sum())}")

    # ── 5. Backtests (all three splits) ──────────────────────────────────────
    print("\n[pipeline] Running TRAIN backtest...")
    train_metrics, train_pf = run_backtest(train_sig, STRATEGY_NAME, "train")

    print("\n[pipeline] Running VAL backtest...")
    val_metrics, val_pf = run_backtest(val_sig, STRATEGY_NAME, "val")

    print("\n[pipeline] Running TEST backtest (out-of-sample)...")
    test_metrics, test_pf = run_backtest(test_sig, STRATEGY_NAME, "test")

    # ── 5b. Cost sensitivity test (2x costs on test) — Phase B3/B4 ───────────
    print("\n[pipeline] Running 2x cost sensitivity test...")
    test_2x_metrics = run_backtest_2x_costs(test_sig, STRATEGY_NAME, "test")
    test_metrics["cost_sensitivity_2x"] = {
        "sharpe_ratio":        test_2x_metrics.get("sharpe_ratio"),
        "annualized_return":   test_2x_metrics.get("annualized_return"),
        "max_drawdown":        test_2x_metrics.get("max_drawdown"),
    }

    # ── 6. Walk-forward (default ON) ─────────────────────────────────────────
    wf_results = None
    if walk_forward:
        print("\n[pipeline] Running walk-forward validation (retrains HMM per window)...")
        wf_results = run_walk_forward(STRATEGY_NAME)
        wf_path = BACKTESTS_DIR / f"{STRATEGY_NAME}_walkforward.json"
        BACKTESTS_DIR.mkdir(exist_ok=True)
        with open(wf_path, "w") as f:
            json.dump(wf_results, f, indent=2)
        if wf_results.get("available"):
            print(f"[pipeline] Walk-forward: mean_sharpe={wf_results.get('mean_sharpe',0):.3f} | positive={wf_results.get('pct_positive',0)*100:.0f}%")

    # ── 7. Hard rejection checks (Phase B2) ──────────────────────────────────
    hrd = hard_rejection_checks(
        metrics             = test_metrics,
        train_regime_dist   = train_regime_dist,
        test_regime_dist    = test_regime_dist,
        cost_sensitivity_metrics = test_2x_metrics,
        walk_forward_run    = (wf_results is not None and wf_results.get("available", False)),
    )
    if hrd["hard_rejected"]:
        print(f"\n[pipeline] HARD REJECTION triggered:")
        for k, v in hrd["rejections"].items():
            print(f"  - {k}: {v}")

    # ── 8. Final multi-split verdict ──────────────────────────────────────────
    mv = multi_split_verdict(
        train_metrics, val_metrics, test_metrics,
        wf_results,
        hard_checks=hrd,
    )
    final_verdict = mv["verdict"]

    # Log flags
    for flag, active in mv["flags"].items():
        if active:
            print(f"[pipeline] FLAG: {flag.replace('_',' ').upper()}")

    # ── 9. Experiment logging (Phase B4) ─────────────────────────────────────
    exp = ExperimentLogger()
    exp.log(
        params        = p,
        config        = CFG,
        train_metrics = train_metrics,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        verdict       = final_verdict,
        wf_results    = wf_results,
        hard_checks   = hrd,
        extra         = {"gap_audit": gap_audit, "dataset": data_ver},
    )

    results = {
        "strategy_name":  STRATEGY_NAME,
        "verdict":        final_verdict,
        "verdict_detail": mv,
        "date":           str(date.today()),
        "experiment_id":  exp.run_id,
        "params":         p,
        "dataset":        data_ver,
        "gap_audit":      gap_audit,
        "regime_distributions": {
            "train": train_regime_dist,
            "test":  test_regime_dist,
        },
        "train_metrics":  train_metrics,
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "walk_forward":   wf_results,
        "next_iteration": _suggest_improvements(test_metrics, mv.get("test_pass", {})),
    }

    # Save final verdict
    verdict_path = REPORTS_DIR / "final_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pipeline] Final verdict saved to {verdict_path}")

    # ── 10. Auto-export if APPROVED (Phase D2) ───────────────────────────────
    auto_save = CFG.get("acceptance", {}).get("auto_save", {})
    ann_ret = test_metrics.get("annualized_return", 0)
    sharpe  = test_metrics.get("sharpe_ratio", 0)
    max_dd  = test_metrics.get("max_drawdown", -999)

    if final_verdict == "APPROVED":
        export_dir = export_approved_strategy(
            results       = results,
            config        = CFG,
            model_path    = MODEL_DIR / "hmm_regime_model.pkl",
            strategy_name = STRATEGY_NAME,
        )
        results["exported_to"] = str(export_dir)
        print(f"\n[pipeline] Strategy APPROVED and EXPORTED to {export_dir}")
    elif (ann_ret >= auto_save.get("min_annualized_return", 0.50) and
          sharpe  >= auto_save.get("min_sharpe", 1.5) and
          max_dd  >= auto_save.get("min_drawdown", -0.25)):
        # Auto-save threshold met but not fully approved — partial export
        export_dir = export_approved_strategy(
            results       = results,
            config        = CFG,
            model_path    = MODEL_DIR / "hmm_regime_model.pkl",
            strategy_name = f"{STRATEGY_NAME}_candidate",
        )
        results["exported_to"] = str(export_dir)
        print(f"\n[pipeline] Candidate strategy exported (auto-save thresholds met): {export_dir}")

    # Summary table
    _print_summary(train_metrics, val_metrics, test_metrics, final_verdict, mv,
                   results.get("exported_to"))

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


def _print_summary(
    train_m: dict, val_m: dict, test_m: dict,
    v: str, mv: dict, exported_to: str = None
) -> None:
    """Print final results across all splits."""
    SEP  = "=" * 65
    DASH = "-" * 65

    def _r(m, k, pct=False, default=0):
        val = m.get(k, default)
        return f"{val*100:>8.2f}%" if pct else f"{val:>8.3f}"

    print(f"\n{SEP}")
    print(f"  XAUUSD SMC+HMM  |  MULTI-SPLIT RESULTS")
    print(f"{SEP}")
    print(f"  {'Metric':<26} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
    print(f"{DASH}")
    print(f"  {'Annualized Return':<26} {_r(train_m,'annualized_return',True):>10} {_r(val_m,'annualized_return',True):>10} {_r(test_m,'annualized_return',True):>10}")
    print(f"  {'Sharpe Ratio':<26} {_r(train_m,'sharpe_ratio'):>10} {_r(val_m,'sharpe_ratio'):>10} {_r(test_m,'sharpe_ratio'):>10}")
    print(f"  {'Max Drawdown':<26} {_r(train_m,'max_drawdown',True):>10} {_r(val_m,'max_drawdown',True):>10} {_r(test_m,'max_drawdown',True):>10}")
    print(f"  {'Profit Factor':<26} {_r(train_m,'profit_factor'):>10} {_r(val_m,'profit_factor'):>10} {_r(test_m,'profit_factor'):>10}")
    print(f"  {'Total Trades':<26} {train_m.get('total_trades','N/A'):>10} {val_m.get('total_trades','N/A'):>10} {test_m.get('total_trades','N/A'):>10}")
    print(f"  {'Win Rate':<26} {_r(train_m,'win_rate',True):>10} {_r(val_m,'win_rate',True):>10} {_r(test_m,'win_rate',True):>10}")
    if "benchmark_return" in test_m:
        print(f"  {'vs Buy & Hold':<26} {'':>10} {'':>10} {_r(test_m,'alpha_vs_benchmark',True):>10}")

    # Cost sensitivity
    cs = test_m.get("cost_sensitivity_2x", {})
    if cs:
        print(f"  {'Sharpe @ 2x costs':<26} {'':>10} {'':>10} {cs.get('sharpe_ratio', 0):>10.3f}")

    print(f"{DASH}")
    print(f"  Split OK:  train={mv['train_ok']}  val={mv['val_ok']}  test={mv['test_ok']}  wf={mv['wf_ok']}")

    # Hard rejections
    hrd = mv.get("hard_checks", {})
    if hrd.get("hard_rejected"):
        for reason in hrd.get("rejections", {}).values():
            print(f"  ! HARD REJECT: {reason}")

    for flag, active in mv["flags"].items():
        if active:
            print(f"  ! FLAG: {flag.replace('_',' ').upper()}")
    print(f"{SEP}")
    print(f"  VERDICT: {v}")
    if exported_to:
        print(f"  Exported to: {exported_to}")
    print(f"{SEP}\n")


def optimize(n_trials: int = 100) -> dict:
    """
    Optuna hyperparameter optimization on VAL data only.
    Test set is never used during optimization.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[pipeline] Optuna not installed. Run: pip install optuna")
        return {}

    train_raw, val_raw, _ = load_split("1H")
    opt_cfg = CFG.get("optimization", {}).get("search_space", {})

    def objective(trial):
        params = {
            "dc_period":             trial.suggest_int("dc_period",         *opt_cfg.get("dc_period",        [10, 50])),
            "adx_threshold":         trial.suggest_float("adx_threshold",   *opt_cfg.get("adx_threshold",    [15.0, 35.0])),
            "hmm_min_prob":          trial.suggest_float("hmm_min_prob",     *opt_cfg.get("hmm_min_prob",     [0.40, 0.80])),
            "atr_stop_mult":         trial.suggest_float("atr_stop_mult",    *opt_cfg.get("atr_stop_mult",    [1.0, 2.5])),
            "atr_tp_mult":           trial.suggest_float("atr_tp_mult",      *opt_cfg.get("atr_tp_mult",      [2.0, 5.0])),
            "atr_expansion_filter":  trial.suggest_categorical("atr_expansion_filter", [True, False]),
            "use_smc":               True,
        }

        try:
            feat   = add_features(val_raw, dc_period=params["dc_period"])
            X, idx = get_hmm_feature_matrix(feat)
            if len(X) < 50:
                return -1e9

            model  = XAUUSDRegimeModel.load("hmm_regime_model")
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
            return sharpe + ret
        except Exception:
            return -1e9

    print(f"[pipeline] Optimizing {n_trials} trials on validation set...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**_build_params_from_config(), **study.best_params}
    print(f"\n[pipeline] Best params: {study.best_params}")
    print(f"[pipeline] Best value: {study.best_value:.4f}")

    best_path = REPORTS_DIR / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD SMC + HMM Strategy Pipeline")
    parser.add_argument("--optimize",          action="store_true", help="Run Optuna optimization first")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward (not recommended)")
    parser.add_argument("--trials",            type=int, default=100, help="Optuna trial count")
    args = parser.parse_args()

    params = None
    if args.optimize:
        params = optimize(n_trials=args.trials)

    results = run_pipeline(
        params       = params,
        walk_forward = not args.skip_walk_forward,
    )
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)
