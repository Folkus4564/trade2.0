"""
app/full_scheme_search_v2.py - Brute-force batch 2: 20 new ideas x 2 targets = 40 experiments.

New ideas explore: long_only, trailing stops, CDC strategy, session variants (London/NY),
high-confidence gating, trend-only router, and idea_16 combinations.

Results saved to artefacts/full_scheme_search_results_v2.json.
Also merges into full_scheme_search_results.json for Streamlit dashboard.

Usage:
    full_scheme_search_v2
    full_scheme_search_v2 --trials 50 --top-wf 3
    full_scheme_search_v2 --ideas 1,2,3
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
# 20 new experiment ideas (batch 2): (name, config_override_dict)
# ---------------------------------------------------------------------------

IDEA_OVERRIDES = [
    # 21: Long-only bias (gold secular uptrend)
    ("idea_21_long_only", {
        "strategy": {"long_only": True},
    }),

    # 22: 4H regime + long-only
    ("idea_22_4h_long_only", {
        "strategy": {"regime_timeframe": "4H", "long_only": True},
    }),

    # 23: London session only (7-16 UTC, tightest spread window)
    ("idea_23_london_session", {
        "session": {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        },
    }),

    # 24: NY session only (13-22 UTC, highest volume)
    ("idea_24_ny_session", {
        "session": {"enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        },
    }),

    # 25: Trailing stop on trend sub-strategy
    ("idea_25_trailing_trend", {
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 26: 4H regime + trailing stop
    ("idea_26_4h_trailing", {
        "strategy": {"regime_timeframe": "4H"},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 27: High confidence gating (only very high-conviction signals)
    ("idea_27_high_confidence", {
        "hmm": {"min_confidence": 0.60},
        "strategies": {
            "trend": {"min_prob": 0.80},
            "range": {"min_prob": 0.65},
        },
    }),

    # 28: Trend-only router (disable range + volatile sub-strategies)
    ("idea_28_trend_only", {
        "strategies": {
            "range":    {"enabled": False},
            "volatile": {"enabled": False},
        },
    }),

    # 29: CDC strategy enabled (Action Zone crossover)
    ("idea_29_cdc_enabled", {
        "strategies": {
            "cdc": {"enabled": True},
        },
    }),

    # 30: No transition cooldown (catch every regime flip)
    ("idea_30_no_cooldown", {
        "regime": {"transition_cooldown_bars": 0},
    }),

    # 31: Long cooldown + high persistence (only very stable regimes)
    ("idea_31_long_cooldown", {
        "regime": {"transition_cooldown_bars": 4, "persistence_bars": 5},
    }),

    # 32: idea_16 + trailing stop (best batch-1 idea + trail)
    ("idea_32_idea16_trailing", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 33: idea_16 + long-only (4H bull regime, longs only)
    ("idea_33_idea16_long_only", {
        "strategy": {"regime_timeframe": "4H", "long_only": True},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
    }),

    # 34: idea_16 + high confidence (filter to very clean setups)
    ("idea_34_idea16_high_conf", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm":      {"n_states": 2, "sizing_max": 2.0, "min_confidence": 0.60},
        "risk":     {"base_allocation_frac": 0.90},
    }),

    # 35: idea_16 + London session only
    ("idea_35_idea16_london", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
        "session":  {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "range":    {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "volatile": {"session_enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        },
    }),

    # 36: idea_16 + trend-only (no range/volatile noise)
    ("idea_36_idea16_trend_only", {
        "strategy": {"regime_timeframe": "4H"},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
        "strategies": {
            "range":    {"enabled": False},
            "volatile": {"enabled": False},
        },
    }),

    # 37: 4H regime + CDC enabled (LuxAlgo Action Zone on 4H view)
    ("idea_37_4h_cdc", {
        "strategy": {"regime_timeframe": "4H"},
        "strategies": {
            "cdc": {"enabled": True},
        },
    }),

    # 38: Tight trailing stop (0.8 ATR — lock in gains faster)
    ("idea_38_tight_trail", {
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 0.8, "atr_tp_mult": 0.0},
        },
    }),

    # 39: idea_16 + long-only + trailing (gold bull + trail protection)
    ("idea_39_idea16_long_trail", {
        "strategy": {"regime_timeframe": "4H", "long_only": True},
        "hmm":      {"n_states": 2, "sizing_max": 2.0},
        "risk":     {"base_allocation_frac": 0.90},
        "strategies": {
            "trend": {"trailing_enabled": True, "trailing_atr_mult": 1.5},
        },
    }),

    # 40: Ultimate combo (4H + 2state + agg + long-only + trailing + London)
    ("idea_40_ultimate", {
        "strategy": {"regime_timeframe": "4H", "long_only": True},
        "hmm":      {"n_states": 2, "sizing_max": 2.0, "min_confidence": 0.55},
        "risk":     {"base_allocation_frac": 0.90},
        "session":  {"enabled": True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        "strategies": {
            "trend":    {
                "trailing_enabled": True, "trailing_atr_mult": 1.5,
                "session_enabled":  True, "allowed_hours_utc": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "range":    {
                "enabled": False,
            },
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
    parser = argparse.ArgumentParser(description="Scheme search batch 2: 20 ideas x 2 targets = 40 experiments")
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
    print(f"  FULL SCHEME SEARCH  --  BATCH 2")
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

    # Save v2 results
    v2_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results_v2.json"
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v2_path, "w") as f:
        json.dump([{"name": n, **r} for n, r in results], f, indent=2, default=str)
    print(f"[search_v2] Batch-2 results saved to {v2_path}")

    # Merge into main results file so Streamlit sees all 80 experiments
    if not args.no_merge:
        main_path   = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
        existing    = _load_existing(main_path)
        new_records = [{"name": n, **r} for n, r in results]
        # Remove any old v2 idea records to allow re-runs without duplication
        v2_names    = {r["idea_name"] for _, r in results}
        existing    = [e for e in existing if e.get("idea_name") not in v2_names]
        merged      = existing + new_records
        with open(main_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"[search_v2] Merged into {main_path} ({len(merged)} total experiments)")

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

            wf_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_v2_wf_results.json"
            with open(wf_path, "w") as f:
                json.dump([{"name": n, **r} for n, r in wf_results], f, indent=2, default=str)
            print(f"[search_v2] WF results saved to {wf_path}")

    total_elapsed = time.time() - start_time
    print(f"\n[search_v2] Total time: {total_elapsed/60:.1f} minutes")
    print(f"[search_v2] Experiments: {total} | Passing DD filter: {len(passing)} | Ranked: {len(ranked)}")

    if ranked:
        best_name, best_result = ranked[0]
        best_tm = best_result.get("test_metrics", {})
        print(f"\n[search_v2] BEST RESULT: {best_name}")
        print(f"  Return : {_safe_pct(best_tm.get('annualized_return'))}")
        print(f"  Sharpe : {_safe_f(best_tm.get('sharpe_ratio'))}")
        print(f"  MaxDD  : {_safe_pct(best_tm.get('max_drawdown'))}")
        print(f"  Trades : {best_tm.get('total_trades', 'N/A')}")
        print(f"  Verdict: {best_result.get('verdict')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
