"""
app/run_pipeline.py - CLI entry point and thin orchestrator.
Replaces both src/pipeline.py and src_v2/pipeline.py.

Usage:
    trade2 --config configs/xauusd_mtf.yaml --retrain-model
    trade2 --skip-walk-forward
    trade2 --optimize --trials 200
    trade2 --retrain-model --export-approved
"""

import argparse
import json
import sys
import numpy as np
from datetime import date
from pathlib import Path

from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from trade2.data.loader import dataset_version
from trade2.data.validation import audit_missing_bars
from trade2.features.builder import add_1h_features, add_5m_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime
from trade2.signals.generator import generate_signals, compute_stops
from trade2.backtesting.engine import run_backtest, run_backtest_2x_costs, run_walk_forward
from trade2.evaluation.hard_rejection import hard_rejection_checks
from trade2.evaluation.verdict import multi_split_verdict
from trade2.experiment.logger import ExperimentLogger
from trade2.export.exporter import export_approved_strategy
from trade2.optimization.optimizer import run_optimization

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/
DATA_ROOT    = Path(__file__).parents[4]  # trade2.0/ (where data/ lives)


def _resolve_artefact_dirs(config: dict) -> dict:
    """Resolve artefact directory paths from config, relative to project root."""
    art_cfg = config.get("artefacts", {})
    root    = PROJECT_ROOT / art_cfg.get("root", "artefacts")
    return {
        "root":       root,
        "models":     PROJECT_ROOT / art_cfg.get("models",      "artefacts/models"),
        "backtests":  PROJECT_ROOT / art_cfg.get("backtests",   "artefacts/backtests"),
        "reports":    PROJECT_ROOT / art_cfg.get("reports",     "artefacts/reports"),
        "experiments": PROJECT_ROOT / art_cfg.get("experiments","artefacts/experiments"),
        "approved":   PROJECT_ROOT / art_cfg.get("approved",    "artefacts/approved_strategies"),
    }


def _regime_distribution(labels) -> dict:
    import pandas as pd
    s = pd.Series(labels.values if hasattr(labels, "values") else list(labels))
    counts = s.value_counts(normalize=True)
    return {regime: round(float(frac), 4) for regime, frac in counts.items()}


def _build_params(config: dict) -> dict:
    feat = config["features"]
    smc  = config["smc"]
    risk = config["risk"]
    hmm  = config["hmm"]
    reg  = config["regime"]
    return {
        "hma_period":              feat["hma_period"],
        "ema_period":              feat["ema_period"],
        "atr_period":              feat["atr_period"],
        "rsi_period":              feat["rsi_period"],
        "adx_period":              feat["adx_period"],
        "dc_period":               feat["dc_period"],
        "adx_threshold":           reg["adx_threshold"],
        "hmm_min_prob":            hmm["min_prob_hard"],
        "hmm_states":              hmm["n_states"],
        "regime_persistence_bars": reg["persistence_bars"],
        "atr_stop_mult":           risk["atr_stop_mult"],
        "atr_tp_mult":             risk["atr_tp_mult"],
        "require_smc_confluence":  smc["require_confluence"],
        "require_pin_bar":         smc["require_pin_bar"],
    }


def run_pipeline(
    config: dict,
    params: dict = None,
    walk_forward: bool = True,
    retrain_model: bool = False,
    export_approved: bool = False,
    optimize: bool = False,
    n_trials: int = 100,
) -> dict:
    """
    Full research pipeline:
    1. Load data (1H always; 5M if multi_tf mode)
    2. Build features
    3. Train/load HMM
    4. Forward-fill regime (multi_tf only)
    5. Generate signals
    6. Backtest all splits + 2x cost sensitivity
    7. Walk-forward validation
    8. Hard rejection checks
    9. Multi-split verdict
    10. Experiment logging
    11. Export if approved + flag set
    """
    p      = {**_build_params(config), **(params or {})}
    dirs   = _resolve_artefact_dirs(config)
    mode   = config.get("strategy", {}).get("mode", "multi_tf")
    strategy_name = config.get("strategy", {}).get("name", "xauusd_mtf_hmm_smc")

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.get("hmm", {}).get("random_seed", 42))

    print(f"\n{'='*60}")
    print(f"  XAUUSD STRATEGY PIPELINE  |  mode={mode}")
    print(f"{'='*60}")
    print(f"  Strategy    : {strategy_name}")
    print(f"  HMM prob    : {p['hmm_min_prob']}  |  states: {p['hmm_states']}")
    print(f"  ATR SL/TP   : {p['atr_stop_mult']}x / {p['atr_tp_mult']}x")
    print(f"  Confluence  : {p['require_smc_confluence']}")
    print(f"{'='*60}\n")

    # ---- 1. Load data ----
    print("[pipeline] Loading 1H data...")
    train_1h, val_1h, test_1h = load_split_tf("1H", config)

    raw_1h_path = DATA_ROOT / config.get("data", {}).get("raw_1h_csv", "data/raw/XAUUSD_1H_2019_2025.csv")
    gap_audit  = audit_missing_bars(train_1h, "1h", config)
    data_ver   = dataset_version(raw_1h_path) if raw_1h_path.exists() else {}
    print(f"  [data] Missing bars: {gap_audit['missing_bars']} ({gap_audit['missing_pct']}%)")

    if mode == "multi_tf":
        print("[pipeline] Loading 5M data...")
        train_5m, val_5m, test_5m = load_split_tf("5M", config)
        print(f"  1H  train={len(train_1h)} | val={len(val_1h)} | test={len(test_1h)}")
        print(f"  5M  train={len(train_5m)} | val={len(val_5m)} | test={len(test_5m)}")
    else:
        print(f"  1H  train={len(train_1h)} | val={len(val_1h)} | test={len(test_1h)}")

    # ---- 2. Build features ----
    print("[pipeline] Engineering 1H features...")
    train_1h_feat = add_1h_features(train_1h, config)
    val_1h_feat   = add_1h_features(val_1h,   config)
    test_1h_feat  = add_1h_features(test_1h,  config)

    if mode == "multi_tf":
        print("[pipeline] Engineering 5M features (SMC)...")
        train_5m_feat = add_5m_features(train_5m, config, dc_period=p["dc_period"])
        val_5m_feat   = add_5m_features(val_5m,   config, dc_period=p["dc_period"])
        test_5m_feat  = add_5m_features(test_5m,  config, dc_period=p["dc_period"])
        for col in ["ob_bullish","ob_bearish","fvg_bullish","fvg_bearish","sweep_low","sweep_high"]:
            if col in train_5m_feat.columns:
                print(f"  [SMC 5M] {col}: {int(train_5m_feat[col].sum())} active bars (train)")
    else:
        for col in ["ob_bullish","ob_bearish","fvg_bullish","fvg_bearish","sweep_low","sweep_high"]:
            if col in train_1h_feat.columns:
                print(f"  [SMC 1H] {col}: {int(train_1h_feat[col].sum())} active bars (train)")

    # ---- 3. Train / load HMM ----
    model_path = dirs["models"] / "hmm_regime_model.pkl"
    if retrain_model or not model_path.exists():
        print("[pipeline] Training HMM regime model...")
        X_train, _ = get_hmm_feature_matrix(train_1h_feat)
        hmm = XAUUSDRegimeModel(
            n_states    = p["hmm_states"],
            random_seed = config.get("hmm", {}).get("random_seed", 42),
        )
        hmm.fit(X_train)
        hmm.save(model_path)
        hmm.summary(X_train)
    else:
        print(f"[pipeline] Loading existing HMM model from {model_path}")
        hmm = XAUUSDRegimeModel.load(model_path)

    # ---- 4. Get regime for all 1H splits ----
    def _get_regime(feat_1h):
        X, idx = get_hmm_feature_matrix(feat_1h)
        return hmm.regime_labels(X), hmm.bull_probability(X), hmm.bear_probability(X), X, idx

    train_labels, train_bull, train_bear, X_train_1h, idx_train_1h = _get_regime(train_1h_feat)
    val_labels,   val_bull,   val_bear,   X_val_1h,   idx_val_1h   = _get_regime(val_1h_feat)
    test_labels,  test_bull,  test_bear,  X_test_1h,  idx_test_1h  = _get_regime(test_1h_feat)

    train_regime_dist = _regime_distribution(train_labels)
    test_regime_dist  = _regime_distribution(test_labels)
    print(f"  [regime] Train: {train_regime_dist}")
    print(f"  [regime] Test : {test_regime_dist}")

    # State distribution from HMM
    dist = hmm.state_distribution(X_train_1h)
    if config.get("hmm", {}).get("n_states", 3) == 3:
        if dist.get("sideways", 0) < 0.10:
            print("[pipeline] WARNING: sideways < 10% of bars")

    # ---- 5. Build signal dataframes ----
    if mode == "multi_tf":
        # Forward-fill 1H regime onto 5M bars.
        # Also forward-fill 1H ATR so compute_stops uses 1H-scale risk levels.
        print("[pipeline] Forward-filling 1H regime onto 5M bars...")
        train_sig_df = forward_fill_1h_regime(train_5m_feat, train_labels, train_bull, train_bear, idx_train_1h,
                                               atr_1h=train_1h_feat["atr_14"].rename(None),
                                               hma_rising=train_1h_feat["hma_rising"].rename(None),
                                               price_above_hma=train_1h_feat["price_above_hma"].rename(None))
        val_sig_df   = forward_fill_1h_regime(val_5m_feat,   val_labels,   val_bull,   val_bear,   idx_val_1h,
                                               atr_1h=val_1h_feat["atr_14"].rename(None),
                                               hma_rising=val_1h_feat["hma_rising"].rename(None),
                                               price_above_hma=val_1h_feat["price_above_hma"].rename(None))
        test_sig_df  = forward_fill_1h_regime(test_5m_feat,  test_labels,  test_bull,  test_bear,  idx_test_1h,
                                               atr_1h=test_1h_feat["atr_14"].rename(None),
                                               hma_rising=test_1h_feat["hma_rising"].rename(None),
                                               price_above_hma=test_1h_feat["price_above_hma"].rename(None))
        freq         = "5min"
    else:
        train_sig_df = train_1h_feat
        val_sig_df   = val_1h_feat
        test_sig_df  = test_1h_feat
        freq         = "1h"

    # Generate signals
    print("[pipeline] Generating signals...")
    sig_kwargs = dict(
        config                  = config,
        adx_threshold           = p["adx_threshold"],
        hmm_min_prob            = p["hmm_min_prob"],
        regime_persistence_bars = p["regime_persistence_bars"],
        require_smc_confluence  = p["require_smc_confluence"],
        require_pin_bar         = p["require_pin_bar"],
    )

    if mode == "multi_tf":
        train_sig = generate_signals(train_sig_df, **sig_kwargs)
        val_sig   = generate_signals(val_sig_df,   **sig_kwargs)
        test_sig  = generate_signals(test_sig_df,  **sig_kwargs)
    else:
        train_sig = generate_signals(
            train_sig_df, **sig_kwargs,
            hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_1h,
        )
        val_sig   = generate_signals(
            val_sig_df, **sig_kwargs,
            hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_1h,
        )
        test_sig  = generate_signals(
            test_sig_df, **sig_kwargs,
            hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_1h,
        )

    train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"])
    val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"])
    test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"])

    print(f"  Train: long={int(train_sig['signal_long'].sum())} | short={int(train_sig['signal_short'].sum())}")
    print(f"  Test : long={int(test_sig['signal_long'].sum())}  | short={int(test_sig['signal_short'].sum())}")

    # ---- 5b. Optional: Optuna optimization on val ----
    if optimize:
        print(f"\n[pipeline] Running Optuna optimization ({n_trials} trials) targeting val_sharpe...")
        best_params, best_val_sharpe = run_optimization(val_sig_df, config, n_trials=n_trials)

        # Override p with best params
        if best_val_sharpe > -990:
            p.update(best_params)
            print(f"[pipeline] Applying best params (val_sharpe={best_val_sharpe:.4f})")

            # Regenerate signals + stops with best params
            sig_kwargs_opt = dict(
                config                  = config,
                adx_threshold           = p.get("adx_threshold",           p["adx_threshold"]),
                hmm_min_prob            = p.get("hmm_min_prob",             p["hmm_min_prob"]),
                regime_persistence_bars = p.get("regime_persistence_bars", p["regime_persistence_bars"]),
                require_smc_confluence  = p.get("require_smc_confluence",  p["require_smc_confluence"]),
                require_pin_bar         = p.get("require_pin_bar",         p["require_pin_bar"]),
            )
            if mode == "multi_tf":
                train_sig = generate_signals(train_sig_df, **sig_kwargs_opt)
                val_sig   = generate_signals(val_sig_df,   **sig_kwargs_opt)
                test_sig  = generate_signals(test_sig_df,  **sig_kwargs_opt)
            else:
                train_sig = generate_signals(train_sig_df, **sig_kwargs_opt,
                    hmm_labels=train_labels, hmm_bull_prob=train_bull, hmm_bear_prob=train_bear, hmm_index=idx_train_1h)
                val_sig   = generate_signals(val_sig_df,   **sig_kwargs_opt,
                    hmm_labels=val_labels,   hmm_bull_prob=val_bull,   hmm_bear_prob=val_bear,   hmm_index=idx_val_1h)
                test_sig  = generate_signals(test_sig_df,  **sig_kwargs_opt,
                    hmm_labels=test_labels,  hmm_bull_prob=test_bull,  hmm_bear_prob=test_bear,  hmm_index=idx_test_1h)

            train_sig = compute_stops(train_sig, p["atr_stop_mult"], p["atr_tp_mult"])
            val_sig   = compute_stops(val_sig,   p["atr_stop_mult"], p["atr_tp_mult"])
            test_sig  = compute_stops(test_sig,  p["atr_stop_mult"], p["atr_tp_mult"])
            print(f"  Optimized: long={int(test_sig['signal_long'].sum())} | short={int(test_sig['signal_short'].sum())} (test)")

    # ---- 6. Backtests ----
    print("\n[pipeline] Running TRAIN backtest...")
    train_metrics, _ = run_backtest(train_sig, strategy_name, "train", config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running VAL backtest...")
    val_metrics,   _ = run_backtest(val_sig,   strategy_name, "val",   config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running TEST backtest (out-of-sample)...")
    test_metrics,  _ = run_backtest(test_sig,  strategy_name, "test",  config, dirs["backtests"], freq=freq)

    print("\n[pipeline] Running 2x cost sensitivity test...")
    test_2x_metrics = run_backtest_2x_costs(test_sig, strategy_name, "test", config, dirs["backtests"], freq=freq)
    test_metrics["cost_sensitivity_2x"] = {
        "sharpe_ratio":      test_2x_metrics.get("sharpe_ratio"),
        "annualized_return": test_2x_metrics.get("annualized_return"),
        "max_drawdown":      test_2x_metrics.get("max_drawdown"),
    }

    # ---- 7. Walk-forward ----
    wf_results = None
    if walk_forward:
        print("\n[pipeline] Running walk-forward validation...")
        wf_results = run_walk_forward(
            strategy_name, config, raw_1h_path, dirs["backtests"], freq="1h"
        )
        if wf_results.get("available"):
            print(f"[pipeline] Walk-forward: mean_sharpe={wf_results.get('mean_sharpe',0):.3f} | positive={wf_results.get('pct_positive',0)*100:.0f}%")

    # ---- 8. Hard rejection checks ----
    hrd = hard_rejection_checks(
        metrics                  = test_metrics,
        train_regime_dist        = train_regime_dist,
        test_regime_dist         = test_regime_dist,
        cost_sensitivity_metrics = test_2x_metrics,
        walk_forward_run         = (wf_results is not None and wf_results.get("available", False)),
    )
    if hrd["hard_rejected"]:
        print(f"\n[pipeline] HARD REJECTION triggered:")
        for k, v in hrd["rejections"].items():
            print(f"  - {k}: {v}")

    # ---- 9. Multi-split verdict ----
    mv            = multi_split_verdict(train_metrics, val_metrics, test_metrics, config, wf_results, hrd)
    final_verdict = mv["verdict"]

    for flag, active in mv["flags"].items():
        if active:
            print(f"[pipeline] FLAG: {flag.replace('_',' ').upper()}")

    # ---- 10. Experiment logging ----
    exp = ExperimentLogger(dirs["experiments"])
    exp.log(
        params        = p,
        config        = config,
        train_metrics = train_metrics,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        verdict       = final_verdict,
        wf_results    = wf_results,
        hard_checks   = hrd,
        extra         = {"gap_audit": gap_audit, "dataset": data_ver},
    )

    results = {
        "strategy_name":  strategy_name,
        "verdict":        final_verdict,
        "verdict_detail": mv,
        "date":           str(date.today()),
        "experiment_id":  exp.run_id,
        "params":         p,
        "dataset":        data_ver,
        "gap_audit":      gap_audit,
        "regime_distributions": {"train": train_regime_dist, "test": test_regime_dist},
        "train_metrics":  train_metrics,
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "walk_forward":   wf_results,
    }

    # Save final verdict JSON
    dirs["reports"].mkdir(parents=True, exist_ok=True)
    verdict_path = dirs["reports"] / "final_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[pipeline] Final verdict saved to {verdict_path}")

    # ---- 11. Export if approved + flag set ----
    if export_approved and final_verdict == "APPROVED":
        export_dir = export_approved_strategy(
            results       = results,
            config        = config,
            artefacts_dir = dirs["root"],
            model_path    = model_path,
            strategy_name = strategy_name,
        )
        results["exported_to"] = str(export_dir)
        print(f"\n[pipeline] Strategy APPROVED and EXPORTED to {export_dir}")

    _print_summary(train_metrics, val_metrics, test_metrics, final_verdict, mv)
    return results


def _print_summary(train_m, val_m, test_m, v, mv):
    SEP  = "=" * 65
    DASH = "-" * 65

    def _r(m, k, pct=False):
        val = m.get(k, 0)
        return f"{val*100:>8.2f}%" if pct else f"{val:>8.3f}"

    print(f"\n{SEP}")
    print(f"  XAUUSD STRATEGY  |  MULTI-SPLIT RESULTS")
    print(f"{SEP}")
    print(f"  {'Metric':<26} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
    print(f"{DASH}")
    print(f"  {'Annualized Return':<26} {_r(train_m,'annualized_return',True):>10} {_r(val_m,'annualized_return',True):>10} {_r(test_m,'annualized_return',True):>10}")
    print(f"  {'Sharpe Ratio':<26} {_r(train_m,'sharpe_ratio'):>10} {_r(val_m,'sharpe_ratio'):>10} {_r(test_m,'sharpe_ratio'):>10}")
    print(f"  {'Max Drawdown':<26} {_r(train_m,'max_drawdown',True):>10} {_r(val_m,'max_drawdown',True):>10} {_r(test_m,'max_drawdown',True):>10}")
    print(f"  {'Profit Factor':<26} {_r(train_m,'profit_factor'):>10} {_r(val_m,'profit_factor'):>10} {_r(test_m,'profit_factor'):>10}")
    print(f"  {'Total Trades':<26} {train_m.get('total_trades','N/A'):>10} {val_m.get('total_trades','N/A'):>10} {test_m.get('total_trades','N/A'):>10}")
    print(f"  {'Win Rate':<26} {_r(train_m,'win_rate',True):>10} {_r(val_m,'win_rate',True):>10} {_r(test_m,'win_rate',True):>10}")
    cs = test_m.get("cost_sensitivity_2x", {})
    if cs:
        print(f"  {'Sharpe @ 2x costs':<26} {'':>10} {'':>10} {cs.get('sharpe_ratio', 0):>10.3f}")
    print(f"{DASH}")
    print(f"  Split OK:  train={mv['train_ok']}  val={mv['val_ok']}  test={mv['test_ok']}  wf={mv['wf_ok']}")
    hrd = mv.get("hard_checks", {})
    if hrd.get("hard_rejected"):
        for reason in hrd.get("rejections", {}).values():
            print(f"  ! HARD REJECT: {reason}")
    for flag, active in mv["flags"].items():
        if active:
            print(f"  ! FLAG: {flag.replace('_',' ').upper()}")
    print(f"{SEP}")
    print(f"  VERDICT: {v}")
    print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Strategy Pipeline (trade2)")
    parser.add_argument("--config",             default="configs/xauusd_mtf.yaml", help="Override config path")
    parser.add_argument("--base-config",        default="configs/base.yaml",       help="Base config path")
    parser.add_argument("--retrain-model",      action="store_true",               help="Force HMM retrain")
    parser.add_argument("--skip-walk-forward",  action="store_true",               help="Skip walk-forward validation")
    parser.add_argument("--optimize",           action="store_true",               help="Run Optuna optimization first")
    parser.add_argument("--trials",             type=int, default=100,             help="Optuna trial count")
    parser.add_argument("--export-approved",    action="store_true",               help="Export strategy if APPROVED")
    args = parser.parse_args()

    # Resolve paths relative to project root
    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None

    config = load_config(base_path, override_path)

    results = run_pipeline(
        config          = config,
        walk_forward    = not args.skip_walk_forward,
        retrain_model   = args.retrain_model,
        export_approved = args.export_approved,
        optimize        = args.optimize,
        n_trials        = args.trials,
    )
    sys.exit(0 if results["verdict"] == "APPROVED" else 1)


if __name__ == "__main__":
    main()
