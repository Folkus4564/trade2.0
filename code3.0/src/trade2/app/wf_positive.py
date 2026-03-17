"""
app/wf_positive.py - Run walk-forward on all positive-returning ideas from full_scheme_search.

Loads full_scheme_search_results.json, picks best run per idea where test_return > 0,
runs WF using the saved optimized params (no re-optimization), saves results.

Usage:
    wf_positive
    wf_positive --min-return 0.05
"""
import argparse
import json
import time
import traceback
from copy import deepcopy
from pathlib import Path

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/


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


def _safe_f(v, d=3):
    return "N/A" if v is None else f"{v:.{d}f}"


def main():
    parser = argparse.ArgumentParser(description="WF on all positive-returning ideas")
    parser.add_argument("--base-config",  default="configs/base.yaml")
    parser.add_argument("--results-file", default="artefacts/full_scheme_search_results.json")
    parser.add_argument("--out-file",     default="artefacts/wf_positive_results.json")
    parser.add_argument("--min-return",   type=float, default=0.0,
                        help="Minimum test annualized_return to include (default 0.0 = any positive)")
    args = parser.parse_args()

    base_path   = PROJECT_ROOT / args.base_config
    base_config = load_config(base_path)

    results_path = PROJECT_ROOT / args.results_file
    if not results_path.exists():
        print(f"[wf_positive] ERROR: {results_path} not found. Run full_scheme_search first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    # Best run per idea (by test_return), filter positives
    best_per_idea = {}
    for exp in all_results:
        idea = exp.get("idea_name") or exp.get("name", "?")
        ret  = (exp.get("test_metrics") or {}).get("annualized_return") or -999
        if ret <= args.min_return:
            continue
        if idea not in best_per_idea or ret > best_per_idea[idea]["_ret"]:
            exp["_ret"] = ret
            best_per_idea[idea] = exp

    candidates = sorted(best_per_idea.values(), key=lambda x: x["_ret"], reverse=True)

    print(f"\n{'='*65}")
    print(f"  WF POSITIVE SEARCH")
    print(f"  {len(candidates)} ideas with test_return > {args.min_return*100:.0f}%")
    print(f"{'='*65}")
    for i, exp in enumerate(candidates, 1):
        print(f"  {i:>2}. {exp.get('idea_name','?'):<38} {exp['_ret']*100:.1f}%")
    print()

    wf_results = []
    start_time = time.time()

    for rank, exp in enumerate(candidates, 1):
        idea_name = exp.get("idea_name") or exp.get("name", "?")
        run_name  = exp.get("run_name", idea_name)
        override  = exp.get("config_override") or {}
        params    = exp.get("params")
        elapsed   = time.time() - start_time

        print(f"\n{'#'*65}")
        print(f"  [WF {rank}/{len(candidates)}] {idea_name}  (orig run: {run_name})")
        print(f"  Test return: {exp['_ret']*100:.1f}% | Elapsed: {elapsed/60:.1f}m")
        print(f"{'#'*65}")

        config = _deep_merge(base_config, override)

        try:
            result = run_pipeline(
                config        = config,
                params        = params,
                retrain_model = True,
                walk_forward  = True,
                optimize      = False,
                legacy_signals= True,
                return_trades = True,
            )
            result["run_name"]      = idea_name + "_wf"
            result["idea_name"]     = idea_name
            result["orig_run_name"] = run_name
            result["config_override"] = override

            tm = result.get("test_metrics") or {}
            wf = result.get("walk_forward") or {}
            print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                  f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                  f"wf_sharpe={_safe_f(wf.get('mean_sharpe'))} "
                  f"wf_pct_pos={_safe_pct(wf.get('pct_positive'))} "
                  f"verdict={result.get('verdict')}")

            wf_results.append(result)

        except Exception as e:
            print(f"  !! FAILED: {e}")
            traceback.print_exc()

    # Rank and display
    wf_results_sorted = sorted(
        wf_results,
        key=lambda x: (x.get("test_metrics") or {}).get("annualized_return") or -999,
        reverse=True,
    )

    SEP  = "=" * 95
    DASH = "-" * 95
    print(f"\n{SEP}")
    print(f"  FINAL WF LEADERBOARD")
    print(f"{SEP}")
    print(f"  {'#':<3} {'Idea':<38} {'Return':>8} {'Sharpe':>7} {'DD':>8} {'WFSharpe':>10} {'WFPos%':>8} {'Verdict'}")
    print(f"{DASH}")
    for i, r in enumerate(wf_results_sorted, 1):
        tm  = r.get("test_metrics") or {}
        wf  = r.get("walk_forward") or {}
        print(f"  {i:<3} {r.get('idea_name','?'):<38} "
              f"{_safe_pct(tm.get('annualized_return')):>8} "
              f"{_safe_f(tm.get('sharpe_ratio')):>7} "
              f"{_safe_pct(tm.get('max_drawdown')):>8} "
              f"{_safe_f(wf.get('mean_sharpe')):>10} "
              f"{_safe_pct(wf.get('pct_positive')):>8} "
              f"{r.get('verdict','?')}")
    print(f"{SEP}\n")

    out_path = PROJECT_ROOT / args.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(wf_results_sorted, f, indent=2, default=str)
    print(f"[wf_positive] Results saved to {out_path}")

    total_elapsed = time.time() - start_time
    print(f"[wf_positive] Total time: {total_elapsed/60:.1f} minutes")

    approved = [r for r in wf_results_sorted if r.get("verdict") == "APPROVED"]
    revise   = [r for r in wf_results_sorted if r.get("verdict") == "REVISE"]
    print(f"[wf_positive] APPROVED: {len(approved)} | REVISE: {len(revise)} | "
          f"REJECTED: {len(wf_results_sorted)-len(approved)-len(revise)}")


if __name__ == "__main__":
    main()
