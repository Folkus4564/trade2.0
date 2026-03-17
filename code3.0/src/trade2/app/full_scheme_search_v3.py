"""
app/full_scheme_search_v3.py - Short-only strategy search: 15 ideas x 2 targets = 30 experiments.

Focuses on short-only XAUUSD setups via HMM bear-regime gating.
Ideas explore: pure short-only, 4H regime, session variants, high confidence,
trailing stops, tight stops, and combinations.

Results saved to artefacts/full_scheme_search_results_v3.json.
Also merges into full_scheme_search_results.json for Streamlit dashboard.

Usage:
    full_scheme_search_v3
    full_scheme_search_v3 --trials 50 --top-wf 3
    full_scheme_search_v3 --ideas 1,2,3
"""

import argparse
import json
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/

# ---------------------------------------------------------------------------
# 15 short-only experiment ideas (batch 3)
# ---------------------------------------------------------------------------

IDEA_OVERRIDES = [
    # 41: Pure short-only baseline (1H regime)
    ("idea_41_short_only", {
        "strategy": {"short_only": True},
    }),

    # 42: 4H regime + short-only (coarser bear detection)
    ("idea_42_4h_short_only", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
    }),

    # 43: 2-state HMM + short-only (no sideways state, cleaner bear signal)
    ("idea_43_2state_short_only", {
        "strategy": {"short_only": True},
        "hmm": {"n_states": 2},
    }),

    # 44: 4H + 2-state + short-only (idea_16 base, shorts only)
    ("idea_44_4h_2state_short_only", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90},
    }),

    # 45: Short-only + high bear confidence (only very convinced HMM)
    ("idea_45_short_high_conf", {
        "strategy": {"short_only": True},
        "hmm": {"min_confidence": 0.60, "min_prob_hard_short": 0.70},
    }),

    # 46: 4H + short-only + high confidence
    ("idea_46_4h_short_high_conf", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
        "hmm": {"n_states": 2, "min_confidence": 0.60, "min_prob_hard_short": 0.70},
        "risk": {"base_allocation_frac": 0.90},
    }),

    # 47: Short-only + London session (bear setups during EU session)
    ("idea_47_short_london", {
        "strategy": {"short_only": True},
        "session": {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        },
    }),

    # 48: Short-only + NY session (US sell-offs)
    ("idea_48_short_ny", {
        "strategy": {"short_only": True},
        "session": {"enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        },
    }),

    # 49: Short-only + trailing stop on trend sub-strategy
    ("idea_49_short_trailing", {
        "strategy": {"short_only": True},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 50: Short-only + tight trailing stop (lock in drops faster)
    ("idea_50_short_tight_trail", {
        "strategy": {"short_only": True},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 0.8, "atr_tp_mult": 0.0},
        },
    }),

    # 51: Short-only + trend-only router (no range/volatile noise)
    ("idea_51_short_trend_only", {
        "strategy": {"short_only": True},
        "strategies": {
            "range":    {"enabled": False},
            "volatile": {"enabled": False},
        },
    }),

    # 52: 4H + 2-state + short-only + trailing (idea_16 base + trail protection)
    ("idea_52_4h_2state_short_trail", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 53: 4H + 2-state + short-only + London (EU open bear momentum)
    ("idea_53_4h_short_london", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
        "hmm": {"n_states": 2, "sizing_max": 2.0},
        "risk": {"base_allocation_frac": 0.90},
        "session": {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        },
    }),

    # 54: Short-only + CDC strategy (Action Zone bearish crossovers)
    ("idea_54_short_cdc", {
        "strategy": {"short_only": True},
        "strategies": {
            "cdc": {"enabled": True},
        },
    }),

    # 55: Ultimate short combo (4H + 2state + high_conf + London + trailing + trend-only)
    ("idea_55_short_ultimate", {
        "strategy": {"regime_timeframe": "4H", "short_only": True},
        "hmm": {"n_states": 2, "sizing_max": 2.0, "min_confidence": 0.55, "min_prob_hard_short": 0.65},
        "risk": {"base_allocation_frac": 0.90},
        "session": {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend": {
                "trailing_enabled": True, "trailing_atr_mult": 1.5,
                "session_enabled":  True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "range":    {"enabled": False},
            "volatile": {"enabled": False},
        },
    }),
]


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _safe_pct(v):
    return "N/A" if v is None else f"{v*100:.1f}%"


def _safe_f(v, decimals=3):
    return "N/A" if v is None else f"{v:.{decimals}f}"


def _print_leaderboard(ranked):
    SEP  = "=" * 90
    DASH = "-" * 90
    print(f"\n{SEP}")
    print(f"  LEADERBOARD  (filtered: DD >= -25%, sorted by test return)")
    print(f"{SEP}")
    print(f"  {'#':<3} {'Experiment':<40} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'Verdict':<12}")
    print(f"{DASH}")
    for rank, (name, result) in enumerate(ranked, 1):
        tm     = result.get("test_metrics", {})
        ret    = _safe_pct(tm.get("annualized_return"))
        sharpe = _safe_f(tm.get("sharpe_ratio"))
        dd     = _safe_pct(tm.get("max_drawdown"))
        trades = tm.get("total_trades", "N/A")
        verdict = result.get("verdict", "N/A")
        print(f"  {rank:<3} {name:<40} {ret:>8} {sharpe:>7} {dd:>8} {str(trades):>7} {verdict:<12}")
    print(f"{SEP}\n")


def _load_existing(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="Scheme search batch 3: short-only strategies (15 ideas x 2 targets)")
    parser.add_argument("--config",      default="configs/xauusd_mtf.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--trials",      type=int,   default=100)
    parser.add_argument("--top-wf",      type=int,   default=3)
    parser.add_argument("--dd-filter",   type=float, default=-0.25)
    parser.add_argument("--ideas",       type=str,   default=None,
                        help="Comma-separated 1-based indices to run (e.g. 1,2,3)")
    parser.add_argument("--no-merge",    action="store_true",
                        help="Do not merge results into full_scheme_search_results.json")
    args = parser.parse_args()

    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None
    base_config   = load_config(base_path, override_path)

    ideas_to_run = IDEA_OVERRIDES
    if args.ideas:
        indices      = [int(x) - 1 for x in args.ideas.split(",")]
        ideas_to_run = [IDEA_OVERRIDES[i] for i in indices if 0 <= i < len(IDEA_OVERRIDES)]

    targets = ["val_sharpe", "val_return"]
    total   = len(ideas_to_run) * len(targets)

    print(f"\n{'='*60}")
    print(f"  FULL SCHEME SEARCH  --  BATCH 3 (SHORT-ONLY)")
    print(f"  {len(ideas_to_run)} ideas x {len(targets)} targets = {total} experiments")
    print(f"  Optuna trials per experiment: {args.trials}")
    print(f"  DD filter: >= {args.dd_filter*100:.0f}%")
    print(f"  Top {args.top_wf} re-run with walk-forward")
    print(f"{'='*60}\n")

    results    = []
    exp_num    = 0
    start_time = time.time()

    for idea_name, override in ideas_to_run:
        for target in targets:
            exp_num += 1
            run_name = f"{idea_name}_opt_{target}"
            elapsed  = time.time() - start_time
            eta      = (elapsed / exp_num) * (total - exp_num) if exp_num > 1 else 0

            print(f"\n{'#'*60}")
            print(f"  [{exp_num}/{total}] {run_name}")
            print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print(f"{'#'*60}")

            config = _deep_merge(base_config, override)

            try:
                result = run_pipeline(
                    config         = config,
                    retrain_model  = True,
                    walk_forward   = False,
                    optimize       = True,
                    n_trials       = args.trials,
                    legacy_signals = True,
                    optuna_target  = target,
                )
                result["run_name"]        = run_name
                result["idea_name"]       = idea_name
                result["optuna_target"]   = target
                result["config_override"] = override
                results.append((run_name, result))

                tm = result.get("test_metrics", {})
                print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"dd={_safe_pct(tm.get('max_drawdown'))} "
                      f"trades={tm.get('total_trades','N/A')}")

            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                results.append((run_name, {
                    "run_name":        run_name,
                    "idea_name":       idea_name,
                    "optuna_target":   target,
                    "verdict":         "ERROR",
                    "test_metrics":    {"annualized_return": None, "sharpe_ratio": None, "max_drawdown": None},
                    "config_override": override,
                    "error":           str(e),
                }))

    # Filter + Rank
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

    # Save v3 results
    v3_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results_v3.json"
    v3_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v3_path, "w") as f:
        json.dump([{"name": n, **r} for n, r in results], f, indent=2, default=str)
    print(f"[search_v3] Batch-3 results saved to {v3_path}")

    # Merge into main results file so Streamlit sees all experiments
    if not args.no_merge:
        main_path   = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
        existing    = _load_existing(main_path)
        new_records = [{"name": n, **r} for n, r in results]
        # Remove any old v3 idea records to allow re-runs without duplication
        v3_names    = {r["idea_name"] for _, r in results}
        existing    = [e for e in existing if e.get("idea_name") not in v3_names]
        merged      = existing + new_records
        with open(main_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"[search_v3] Merged into {main_path} ({len(merged)} total experiments)")

    # Re-run top N with WF
    if ranked and args.top_wf > 0:
        print(f"\n{'='*60}")
        print(f"  RE-RUNNING TOP {args.top_wf} WITH WALK-FORWARD VALIDATION")
        print(f"{'='*60}")
        wf_results = []
        for rank, (name, result) in enumerate(ranked[:args.top_wf], 1):
            print(f"\n[WF {rank}/{args.top_wf}] {name}")
            override = result.get("config_override", {})
            config   = _deep_merge(base_config, override)
            try:
                wf_result = run_pipeline(
                    config         = config,
                    params         = result.get("params"),
                    retrain_model  = True,
                    walk_forward   = True,
                    optimize       = False,
                    legacy_signals = True,
                )
                wf_result["run_name"] = name + "_wf"
                wf_results.append((name, wf_result))

                tm = wf_result.get("test_metrics", {})
                wf = wf_result.get("walk_forward", {})
                print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"wf_sharpe={_safe_f(wf.get('mean_sharpe') if wf else None)} "
                      f"verdict={wf_result.get('verdict')}")

            except Exception as e:
                print(f"  !! WF FAILED: {e}")
                traceback.print_exc()

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

            wf_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_v3_wf_results.json"
            with open(wf_path, "w") as f:
                json.dump([{"name": n, **r} for n, r in wf_results], f, indent=2, default=str)
            print(f"[search_v3] WF results saved to {wf_path}")

    total_elapsed = time.time() - start_time
    print(f"\n[search_v3] Total time: {total_elapsed/60:.1f} minutes")
    print(f"[search_v3] Experiments: {total} | Passing DD filter: {len(passing)} | Ranked: {len(ranked)}")

    if ranked:
        best_name, best_result = ranked[0]
        best_tm = best_result.get("test_metrics", {})
        print(f"\n[search_v3] BEST RESULT: {best_name}")
        print(f"  Return : {_safe_pct(best_tm.get('annualized_return'))}")
        print(f"  Sharpe : {_safe_f(best_tm.get('sharpe_ratio'))}")
        print(f"  MaxDD  : {_safe_pct(best_tm.get('max_drawdown'))}")
        print(f"  Trades : {best_tm.get('total_trades', 'N/A')}")
        print(f"  Verdict: {best_result.get('verdict')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
