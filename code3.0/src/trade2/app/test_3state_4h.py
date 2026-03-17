"""
app/test_3state_4h.py - Test idea_16 with 3-state HMM on 4H.

The hypothesis: 2-state HMM labels 2022 bear as "bull" => losing longs.
With 3 states (bull/bear/sideways), the strategy goes flat in sideways
and avoids the 2022 choppy/bear period entirely.

Usage:
    test_3state_4h
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


def _print_wf_windows(wf):
    if not wf or not wf.get("windows"):
        return
    print(f"\n  WF Windows:")
    print(f"  {'Win':<4} {'Period':<35} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'Beats?':>7}")
    print("  " + "-"*78)
    for w in wf["windows"]:
        beats = "YES" if w.get("beats_random_baseline") else "NO"
        print(f"  {w['window']:<4} {w.get('val_period','?'):<35} "
              f"{w.get('annualized_return',0)*100:>7.1f}% "
              f"{w.get('sharpe_ratio',0):>8.3f} "
              f"{w.get('total_trades',0):>7} "
              f"{w.get('win_rate',0)*100:>7.1f}% "
              f"{beats:>7}")


def _run_and_report(label, override, base_config, params=None):
    config = _deep_merge(base_config, override)
    print(f"\n{'='*65}")
    print(f"  TEST: {label}")
    print(f"{'='*65}\n")

    result = run_pipeline(
        config        = config,
        params        = params,
        retrain_model = True,
        walk_forward  = True,
        optimize      = False,
        legacy_signals= True,
        return_trades = True,
    )

    tm = result.get("test_metrics") or {}
    wf = result.get("walk_forward") or {}

    print(f"\n{'='*65}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*65}")
    print(f"  Verdict        : {result.get('verdict')}")
    print(f"  Test Return    : {tm.get('annualized_return', 0)*100:.1f}%")
    print(f"  Test Sharpe    : {tm.get('sharpe_ratio', 0):.3f}")
    print(f"  Test MaxDD     : {tm.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Test Trades    : {tm.get('total_trades', 0)}")
    print(f"  Test Win Rate  : {tm.get('win_rate', 0)*100:.1f}%")
    print(f"  Test PF        : {tm.get('profit_factor', 0):.3f}")
    print(f"  WF Mean Sharpe : {wf.get('mean_sharpe', 'N/A')}")
    print(f"  WF Pct Positive: {wf.get('pct_positive', 0)*100:.0f}%" if wf else "  WF: N/A")
    _print_wf_windows(wf)

    return result


def main():
    base_config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    # Load best params from idea_16 (keep SMC/SL/TP params, reset HMM state count)
    results_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
    base_params = None
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        for exp in all_results:
            if exp.get("idea_name") == "idea_16_4h_2state_agg" and \
               exp.get("optuna_target") == "val_return":
                base_params = exp.get("params")
                break
        if base_params:
            # Force 3-state HMM (override the param)
            base_params = {**base_params, "hmm_states": 3}
            print(f"[test_3state_4h] Using idea_16 params + hmm_states=3")

    # --- Variant A: 4H 3-state, aggressive sizing, long_only=False ---
    result_a = _run_and_report(
        label    = "4H + 3-state HMM + aggressive (both directions)",
        override = {
            "strategy": {"regime_timeframe": "4H"},
            "hmm":      {"n_states": 3, "sizing_max": 2.0},
            "risk":     {"base_allocation_frac": 0.90},
        },
        base_config = base_config,
        params      = base_params,
    )

    # --- Variant B: 4H 3-state, long_only=True ---
    result_b = _run_and_report(
        label    = "4H + 3-state HMM + aggressive + long_only=True",
        override = {
            "strategy": {"regime_timeframe": "4H", "long_only": True},
            "hmm":      {"n_states": 3, "sizing_max": 2.0},
            "risk":     {"base_allocation_frac": 0.90},
        },
        base_config = base_config,
        params      = base_params,
    )

    # Save both results
    out = {
        "3state_4h_both_dirs": result_a,
        "3state_4h_long_only": result_b,
    }
    out_path = PROJECT_ROOT / "artefacts" / "idea16_3state_4h_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[test_3state_4h] Results saved to {out_path}")

    # Final comparison
    def _fmt(r, k, pct=False):
        v = (r.get("test_metrics") or {}).get(k, 0)
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"

    def _wf(r):
        wf = r.get("walk_forward") or {}
        return f"sharpe={wf.get('mean_sharpe', 0):.3f} pos={wf.get('pct_positive', 0)*100:.0f}%"

    print(f"\n{'='*75}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Variant':<45} {'Ret':>7} {'Sharpe':>7} {'DD':>7} {'WF'}")
    print(f"  {'-'*72}")
    print(f"  {'4H 3-state both dirs':<45} {_fmt(result_a,'annualized_return',True):>7} "
          f"{_fmt(result_a,'sharpe_ratio'):>7} {_fmt(result_a,'max_drawdown',True):>7} {_wf(result_a)}")
    print(f"  {'4H 3-state long_only':<45} {_fmt(result_b,'annualized_return',True):>7} "
          f"{_fmt(result_b,'sharpe_ratio'):>7} {_fmt(result_b,'max_drawdown',True):>7} {_wf(result_b)}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
