"""
app/scalp_baseline_sweep.py - Optuna baseline sweep for scalping research.

Calibrates the scalping config before indicator testing begins.
Runs the full pipeline with optimize=True using scalp-specific search bounds.
Optionally compares 1M (scalp.yaml) vs 5M (scalp_5m.yaml) signal timeframes.

Usage:
    scalp_baseline_sweep
    scalp_baseline_sweep --scalp-config configs/scalp_5m.yaml
    scalp_baseline_sweep --trials 200
    scalp_baseline_sweep --compare-tf
    scalp_baseline_sweep --no-retrain
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]   # code3.0/


def _run_sweep(base_config_path: Path, scalp_config_path: Path, trials: int,
               retrain: bool, skip_wf: bool) -> dict:
    """Run the full pipeline with optimization for one config. Returns results dict."""
    from trade2.config.loader import load_config
    from trade2.app.run_pipeline import run_pipeline

    config = load_config(str(base_config_path), str(scalp_config_path))

    # Override n_trials from CLI if provided
    config.setdefault("optimization", {})["n_trials"] = trials

    tf = config.get("strategy", {}).get("signal_timeframe", "?M")
    print(f"\n[sweep] Starting baseline sweep: signal_tf={tf}, trials={trials}")
    print(f"[sweep] Config: {base_config_path.name} + {scalp_config_path.name}")

    results = run_pipeline(
        config              = config,
        walk_forward        = not skip_wf,
        retrain_model       = retrain,
        export_approved     = False,
        optimize            = True,
        n_trials            = trials,
        legacy_signals      = False,
        model_path_override = None,
    )
    return results


def _save_sweep_result(results: dict, out_dir: Path, label: str) -> Path:
    """Save sweep results to artefacts/scalp_research/baseline_sweep_best_{label}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_sweep_best_{label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[sweep] Results saved to {out_path}")
    return out_path


def _print_comparison(r1m: dict, r5m: dict) -> None:
    """Print a side-by-side comparison of 1M vs 5M sweep results."""
    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  SCALP BASELINE SWEEP COMPARISON: 1M vs 5M")
    print(SEP)
    t1m = r1m.get("test_metrics", {})
    t5m = r5m.get("test_metrics", {})

    def _r(m, k, pct=False):
        v = m.get(k, float("nan"))
        try:
            return f"{v*100:>8.2f}%" if pct else f"{v:>8.3f}"
        except (TypeError, ValueError):
            return f"{'N/A':>8}"

    print(f"  {'Metric':<26} {'1M':>12} {'5M':>12}")
    print("-" * 52)
    for label, key, pct in [
        ("Ann. Return",    "annualized_return", True),
        ("Sharpe",         "sharpe_ratio",       False),
        ("Max Drawdown",   "max_drawdown",       True),
        ("Profit Factor",  "profit_factor",      False),
        ("Win Rate",       "win_rate",           True),
        ("Total Trades",   "total_trades",       False),
    ]:
        v1 = _r(t1m, key, pct)
        v5 = _r(t5m, key, pct)
        print(f"  {label:<26} {v1:>12} {v5:>12}")

    print(f"  {'Verdict':<26} {r1m.get('verdict','?'):>12} {r5m.get('verdict','?'):>12}")
    print(SEP)

    # Recommend winner
    sh1 = t1m.get("sharpe_ratio", -999)
    sh5 = t5m.get("sharpe_ratio", -999)
    winner = "5M" if sh5 > sh1 else "1M"
    print(f"\n[sweep] Recommended TF: {winner} (higher Sharpe = {max(sh1, sh5):.3f})")
    if winner == "5M":
        print("[sweep] Proceed with: scalp_research --scalp-config configs/scalp_5m.yaml")
    else:
        print("[sweep] Proceed with: scalp_research --scalp-config configs/scalp.yaml")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna baseline sweep for scalping research (1M and/or 5M)"
    )
    parser.add_argument(
        "--scalp-config", default="configs/scalp.yaml",
        help="Path to scalp config overlay. Default: configs/scalp.yaml"
    )
    parser.add_argument(
        "--base-config", default="configs/base.yaml",
        help="Base config path. Default: configs/base.yaml"
    )
    parser.add_argument(
        "--trials", type=int, default=200,
        help="Optuna trial count. Default: 200"
    )
    parser.add_argument(
        "--compare-tf", action="store_true",
        help="Run both scalp.yaml (1M) and scalp_5m.yaml (5M) and compare results"
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="Skip HMM retrain (reuse existing model)"
    )
    parser.add_argument(
        "--skip-walk-forward", action="store_true",
        help="Skip walk-forward validation (faster)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save sweep results. Default: artefacts/scalp_research"
    )
    args = parser.parse_args()

    base_cfg   = PROJECT_ROOT / args.base_config
    out_dir    = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artefacts" / "scalp_research"
    retrain    = not args.no_retrain
    skip_wf    = args.skip_walk_forward

    if args.compare_tf:
        # Run both 1M and 5M sweeps
        cfg_1m = PROJECT_ROOT / "configs" / "scalp.yaml"
        cfg_5m = PROJECT_ROOT / "configs" / "scalp_5m.yaml"

        print("[sweep] Running 1M sweep...")
        r1m = _run_sweep(base_cfg, cfg_1m, args.trials, retrain, skip_wf)
        _save_sweep_result(r1m, out_dir, "1m")

        print("[sweep] Running 5M sweep...")
        r5m = _run_sweep(base_cfg, cfg_5m, args.trials, retrain=False, skip_wf=skip_wf)
        _save_sweep_result(r5m, out_dir, "5m")

        # Combined summary
        combined = {"1m": r1m, "5m": r5m}
        combined_path = out_dir / "baseline_sweep_comparison.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"[sweep] Comparison saved to {combined_path}")

        _print_comparison(r1m, r5m)
    else:
        # Run single sweep with provided scalp config
        scalp_cfg = PROJECT_ROOT / args.scalp_config if not Path(args.scalp_config).is_absolute() else Path(args.scalp_config)
        tf_label  = "5m" if "5m" in scalp_cfg.name.lower() else "1m"
        results   = _run_sweep(base_cfg, scalp_cfg, args.trials, retrain, skip_wf)
        _save_sweep_result(results, out_dir, tf_label)

    sys.exit(0)


if __name__ == "__main__":
    main()
