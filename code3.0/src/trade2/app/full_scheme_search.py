"""
app/full_scheme_search.py - Brute-force 50% return search.

Runs 20 experimental ideas x 2 Optuna targets (val_sharpe + val_return) = 40 experiments.
Filters results by max_drawdown >= -25%, ranks by test annualized return.
Re-runs top 3 with walk-forward validation.

Usage:
    full_scheme_search
    full_scheme_search --trials 50 --top-wf 3
"""

import argparse
import json
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/


# ---------------------------------------------------------------------------
# 20 experiment ideas: (name, config_override_dict)
# ---------------------------------------------------------------------------

IDEA_OVERRIDES = [
    # 1: Baseline (legacy 1H/5M, no changes)
    ("idea_01_baseline", {}),

    # 2: 4H regime + 5M signals
    ("idea_02_4h_regime", {
        "strategy": {"regime_timeframe": "4H"},
    }),

    # 3: 1H regime + 15M signals
    ("idea_03_15m_signals", {
        "strategy": {"signal_timeframe": "15M"},
    }),

    # 4: 1H regime + 30M signals
    ("idea_04_30m_signals", {
        "strategy": {"signal_timeframe": "30M"},
    }),

    # 5: 4H regime + 15M signals
    ("idea_05_4h_15m", {
        "strategy": {"regime_timeframe": "4H", "signal_timeframe": "15M"},
    }),

    # 6: HMM 2-state (bull/bear only, no sideways)
    ("idea_06_hmm_2state", {
        "hmm": {"n_states": 2},
    }),

    # 7: HMM 4-state (more granular regimes)
    ("idea_07_hmm_4state", {
        "hmm": {"n_states": 4},
    }),

    # 8: Aggressive sizing (lever up existing edge)
    ("idea_08_aggressive_sizing", {
        "risk": {"base_allocation_frac": 0.90},
        "hmm": {"sizing_max": 2.0},
    }),

    # 9: No session filter (24h trading, more signals)
    ("idea_09_no_session", {
        "session": {"enabled": False},
    }),

    # 10: Wide stops, ride trends hard
    ("idea_10_wide_stops", {
        "risk": {"atr_stop_mult": 4.0, "atr_tp_mult": 25.0},
    }),

    # 11: Quick scalp (high-freq mean reversion)
    ("idea_11_quick_scalp", {
        "risk": {"atr_stop_mult": 1.5, "atr_tp_mult": 3.0, "max_hold_bars": 24},
    }),

    # 12: HMM minimal features (3 features, reduce overfitting)
    ("idea_12_hmm_minimal", {
        "hmm": {"features": ["hmm_feat_ret", "hmm_feat_atr", "hmm_feat_vol"]},
    }),

    # 13: 4H + aggressive + wide stops
    ("idea_13_4h_agg_wide", {
        "strategy": {"regime_timeframe": "4H"},
        "risk": {"base_allocation_frac": 0.90, "atr_stop_mult": 4.0, "atr_tp_mult": 25.0},
        "hmm": {"sizing_max": 2.0},
    }),

    # 14: 4H + 15M + no session (slow regime + 24h coverage)
    ("idea_14_4h_15m_nosession", {
        "strategy": {"regime_timeframe": "4H", "signal_timeframe": "15M"},
        "session": {"enabled": False},
    }),

    # 15: 2-state + aggressive + wide
    ("idea_15_2state_agg_wide", {
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90, "atr_stop_mult": 4.0},
    }),

    # 16: 4H + 2-state + aggressive
    ("idea_16_4h_2state_agg", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90},
    }),

    # 17: 15M + no session + aggressive
    ("idea_17_15m_nosession_agg", {
        "strategy": {"signal_timeframe": "15M"},
        "session": {"enabled": False},
        "risk": {"base_allocation_frac": 0.90},
        "hmm": {"sizing_max": 2.0},
    }),

    # 18: 30M + 4-state + wide stops
    ("idea_18_30m_4state_wide", {
        "strategy": {"signal_timeframe": "30M"},
        "hmm": {"n_states": 4},
        "risk": {"atr_stop_mult": 4.0, "atr_tp_mult": 25.0},
    }),

    # 19: 4H + minimal HMM + aggressive
    ("idea_19_4h_minimal_agg", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm": {"features": ["hmm_feat_ret", "hmm_feat_atr", "hmm_feat_vol"], "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90},
    }),

    # 20: Kitchen sink (max everything combined)
    ("idea_20_kitchen_sink", {
        "strategy": {"regime_timeframe": "4H", "signal_timeframe": "15M"},
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "session": {"enabled": False},
        "risk": {"base_allocation_frac": 0.90, "atr_stop_mult": 4.0, "atr_tp_mult": 25.0},
    }),
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a deep copy of base."""
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _safe_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


def _safe_f(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _print_leaderboard(ranked):
    SEP  = "=" * 90
    DASH = "-" * 90
    print(f"\n{SEP}")
    print(f"  LEADERBOARD  (filtered: DD >= -25%, sorted by test return)")
    print(f"{SEP}")
    print(f"  {'#':<3} {'Experiment':<40} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'Verdict':<12}")
    print(f"{DASH}")
    for rank, (name, result) in enumerate(ranked, 1):
        tm = result.get("test_metrics", {})
        ret = _safe_pct(tm.get("annualized_return"))
        sharpe = _safe_f(tm.get("sharpe_ratio"))
        dd = _safe_pct(tm.get("max_drawdown"))
        trades = tm.get("total_trades", "N/A")
        verdict = result.get("verdict", "N/A")
        print(f"  {rank:<3} {name:<40} {ret:>8} {sharpe:>7} {dd:>8} {str(trades):>7} {verdict:<12}")
    print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="Full scheme search: 20 ideas x 2 targets = 40 experiments")
    parser.add_argument("--config",      default="configs/xauusd_mtf.yaml", help="Override config path")
    parser.add_argument("--base-config", default="configs/base.yaml",       help="Base config path")
    parser.add_argument("--trials",      type=int, default=100,             help="Optuna trials per experiment")
    parser.add_argument("--top-wf",      type=int, default=3,               help="Top N to re-run with walk-forward")
    parser.add_argument("--dd-filter",   type=float, default=-0.25,         help="Max drawdown filter (default -0.25)")
    parser.add_argument("--ideas",       type=str, default=None,            help="Comma-separated idea indices to run (e.g. 1,2,3)")
    args = parser.parse_args()

    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None
    base_config   = load_config(base_path, override_path)

    # Filter ideas if --ideas specified
    ideas_to_run = IDEA_OVERRIDES
    if args.ideas:
        indices = [int(x) - 1 for x in args.ideas.split(",")]
        ideas_to_run = [IDEA_OVERRIDES[i] for i in indices if 0 <= i < len(IDEA_OVERRIDES)]

    targets = ["val_sharpe", "val_return"]
    total = len(ideas_to_run) * len(targets)

    print(f"\n{'='*60}")
    print(f"  FULL SCHEME SEARCH")
    print(f"  {len(ideas_to_run)} ideas x {len(targets)} targets = {total} experiments")
    print(f"  Optuna trials per experiment: {args.trials}")
    print(f"  DD filter: >= {args.dd_filter*100:.0f}%")
    print(f"  Top {args.top_wf} re-run with walk-forward")
    print(f"{'='*60}\n")

    results = []
    exp_num = 0
    start_time = time.time()

    for idea_name, override in ideas_to_run:
        for target in targets:
            exp_num += 1
            run_name = f"{idea_name}_opt_{target}"
            elapsed = time.time() - start_time
            eta = (elapsed / exp_num) * (total - exp_num) if exp_num > 1 else 0

            print(f"\n{'#'*60}")
            print(f"  [{exp_num}/{total}] {run_name}")
            print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print(f"{'#'*60}")

            config = _deep_merge(base_config, override)

            try:
                result = run_pipeline(
                    config          = config,
                    retrain_model   = True,
                    walk_forward    = False,
                    optimize        = True,
                    n_trials        = args.trials,
                    legacy_signals  = True,
                    optuna_target   = target,
                )
                result["run_name"]     = run_name
                result["idea_name"]    = idea_name
                result["optuna_target"] = target
                result["config_override"] = override
                results.append((run_name, result))

                tm = result.get("test_metrics", {})
                print(f"  >> Result: return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"dd={_safe_pct(tm.get('max_drawdown'))} "
                      f"trades={tm.get('total_trades','N/A')}")

            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                results.append((run_name, {
                    "run_name": run_name,
                    "idea_name": idea_name,
                    "optuna_target": target,
                    "verdict": "ERROR",
                    "test_metrics": {"annualized_return": None, "sharpe_ratio": None, "max_drawdown": None},
                    "error": str(e),
                }))

    # ----------- Filter + Rank -----------
    dd_filter = args.dd_filter
    passing = [
        (n, r) for n, r in results
        if r.get("test_metrics", {}).get("max_drawdown") is not None
        and r["test_metrics"]["max_drawdown"] >= dd_filter
    ]
    ranked = sorted(
        passing,
        key=lambda x: x[1]["test_metrics"].get("annualized_return") or -999,
        reverse=True,
    )

    _print_leaderboard(ranked)

    # Save full results
    results_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump([{"name": n, **r} for n, r in results], f, indent=2, default=str)
    print(f"[search] Full results saved to {results_path}")

    # ----------- Re-run top N with WF -----------
    if ranked and args.top_wf > 0:
        print(f"\n{'='*60}")
        print(f"  RE-RUNNING TOP {args.top_wf} WITH WALK-FORWARD VALIDATION")
        print(f"{'='*60}")
        wf_results = []
        for rank, (name, result) in enumerate(ranked[:args.top_wf], 1):
            print(f"\n[WF {rank}/{args.top_wf}] {name}")
            override = result.get("config_override", {})
            config = _deep_merge(base_config, override)
            # Apply best params from previous optimization run if available
            try:
                wf_result = run_pipeline(
                    config          = config,
                    params          = result.get("params"),
                    retrain_model   = True,
                    walk_forward    = True,
                    optimize        = False,
                    legacy_signals  = True,
                )
                wf_result["run_name"] = name + "_wf"
                wf_results.append((name, wf_result))

                tm = wf_result.get("test_metrics", {})
                wf = wf_result.get("walk_forward", {})
                print(f"  >> WF result: return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"wf_sharpe={_safe_f(wf.get('mean_sharpe') if wf else None)} "
                      f"verdict={wf_result.get('verdict')}")

            except Exception as e:
                print(f"  !! WF FAILED: {e}")
                traceback.print_exc()

        # Final WF leaderboard
        if wf_results:
            wf_passing = [
                (n, r) for n, r in wf_results
                if r.get("test_metrics", {}).get("max_drawdown") is not None
                and r["test_metrics"]["max_drawdown"] >= dd_filter
            ]
            wf_ranked = sorted(
                wf_passing,
                key=lambda x: x[1]["test_metrics"].get("annualized_return") or -999,
                reverse=True,
            )
            print(f"\n{'='*60}")
            print(f"  FINAL WALK-FORWARD LEADERBOARD")
            _print_leaderboard(wf_ranked)

            wf_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_wf_results.json"
            with open(wf_path, "w") as f:
                json.dump([{"name": n, **r} for n, r in wf_results], f, indent=2, default=str)
            print(f"[search] WF results saved to {wf_path}")

    total_elapsed = time.time() - start_time
    print(f"\n[search] Total time: {total_elapsed/60:.1f} minutes")
    print(f"[search] Experiments run: {total} | Passing DD filter: {len(passing)} | Ranked: {len(ranked)}")

    if ranked:
        best_name, best_result = ranked[0]
        best_tm = best_result.get("test_metrics", {})
        print(f"\n[search] BEST RESULT: {best_name}")
        print(f"  Return: {_safe_pct(best_tm.get('annualized_return'))}")
        print(f"  Sharpe: {_safe_f(best_tm.get('sharpe_ratio'))}")
        print(f"  MaxDD:  {_safe_pct(best_tm.get('max_drawdown'))}")
        print(f"  Trades: {best_tm.get('total_trades', 'N/A')}")
        print(f"  Verdict: {best_result.get('verdict')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
