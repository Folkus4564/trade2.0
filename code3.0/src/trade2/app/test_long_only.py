"""
app/test_long_only.py - Test idea_16_4h_2state_agg with long_only=True + WF.

Usage:
    test_long_only
"""
import json
from copy import deepcopy
from pathlib import Path

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parents[3]


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def main():
    base_config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    # Load best params from previous idea_16 run
    results_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
    best_params = None
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        for exp in all_results:
            if exp.get("idea_name") == "idea_16_4h_2state_agg" and \
               exp.get("optuna_target") == "val_return":
                best_params = exp.get("params")
                break
        print(f"[test_long_only] Loaded best params: {best_params}")

    # idea_16 override + long_only=True
    override = {
        "strategy": {"regime_timeframe": "4H", "long_only": True},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
    }
    config = _deep_merge(base_config, override)

    print("\n" + "="*65)
    print("  TEST: idea_16_4h_2state_agg + long_only=True")
    print("  4H regime | 2-state HMM | aggressive sizing | LONGS ONLY")
    print("="*65 + "\n")

    result = run_pipeline(
        config        = config,
        params        = best_params,
        retrain_model = True,
        walk_forward  = True,
        optimize      = False,
        legacy_signals= True,
        return_trades = True,
    )

    tm = result.get("test_metrics") or {}
    wf = result.get("walk_forward") or {}

    print("\n" + "="*65)
    print("  LONG_ONLY RESULT SUMMARY")
    print("="*65)
    print(f"  Verdict      : {result.get('verdict')}")
    print(f"  Test Return  : {tm.get('annualized_return', 0)*100:.1f}%")
    print(f"  Test Sharpe  : {tm.get('sharpe_ratio', 0):.3f}")
    print(f"  Test MaxDD   : {tm.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Test Trades  : {tm.get('total_trades', 0)}")
    print(f"  Test Win Rate: {tm.get('win_rate', 0)*100:.1f}%")
    print(f"  Test PF      : {tm.get('profit_factor', 0):.3f}")
    print(f"  WF Mean Sharpe : {wf.get('mean_sharpe', 'N/A')}")
    print(f"  WF Pct Positive: {wf.get('pct_positive', 0)*100:.0f}%" if wf else "  WF: N/A")
    print("="*65)

    if wf and wf.get("windows"):
        print("\n  WF Windows:")
        print(f"  {'Win':<4} {'Period':<35} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8}")
        print("  " + "-"*70)
        for w in wf["windows"]:
            print(f"  {w['window']:<4} {w.get('val_period','?'):<35} "
                  f"{w.get('annualized_return',0)*100:>7.1f}% "
                  f"{w.get('sharpe_ratio',0):>8.3f} "
                  f"{w.get('total_trades',0):>7} "
                  f"{w.get('win_rate',0)*100:>7.1f}%")

    # Save result
    out_path = PROJECT_ROOT / "artefacts" / "idea16_long_only_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[test_long_only] Result saved to {out_path}")


if __name__ == "__main__":
    main()
