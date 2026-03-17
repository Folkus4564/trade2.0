"""
app/full_scheme_search_3state.py - 3-state HMM scheme search.

Systematic search for the best 3-state HMM configuration on XAUUSD.
Motivation: 2-state HMM misclassifies 2022 bear market as "bull" (no sideways state
to absorb ambiguous bars), causing walk-forward failures.  3-state adds a sideways
regime that explicitly captures mean-reverting / ranging / bear market conditions.

All 15 ideas force n_states=3.  Key 3-state adjustments vs 2-state:
  - min_prob_hard / min_prob_entry: lowered to ~0.60-0.65 (3-way split, each state ~33%)
  - min_confidence: lowered to ~0.40-0.45 (max-state prob naturally lower)
  - sideways probability now routes to range strategy (non-zero signal)

Ideas x targets: 15 x 2 = 30 experiments total.

Usage:
    full_scheme_search_3state
    full_scheme_search_3state --trials 100 --top-wf 5
    full_scheme_search_3state --ideas 1,3,7     # run specific ideas only
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
# 15 three-state experiment ideas
# ---------------------------------------------------------------------------
# Every idea implicitly forces n_states=3.
# Overrides are merged on top of base.yaml.
# ---------------------------------------------------------------------------

_3STATE_BASE = {
    "hmm": {
        "n_states": 3,
        "min_prob_hard":       0.60,
        "min_prob_hard_short": 0.60,
        "min_prob_entry":      0.65,
        "min_prob_exit":       0.45,
        "min_confidence":      0.45,
        "sizing_max":          2.0,
    },
    "risk": {"base_allocation_frac": 0.90},
    "strategy": {"regime_timeframe": "4H"},
}


def _merge3(base, *overrides):
    """Deep-merge base + all overrides left to right."""
    result = deepcopy(base)
    for ov in overrides:
        for k, v in ov.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = _merge3(result[k], v)
            else:
                result[k] = deepcopy(v)
    return result


IDEA_OVERRIDES = [
    # --- 1. Baseline: idea_16 ported to 3-state (both directions) ---
    ("idea_3s_01_4h_base", _merge3(_3STATE_BASE, {})),

    # --- 2. Long-only: avoids shorts in sideways/bear (gold uptrend bias) ---
    ("idea_3s_02_4h_long_only", _merge3(_3STATE_BASE, {
        "strategy": {"long_only": True},
    })),

    # --- 3. Drawdown filter: bear market circuit breaker ---
    ("idea_3s_03_4h_dd_filter", _merge3(_3STATE_BASE, {
        "drawdown_filter": {
            "enabled":           True,
            "lookback_bars":     120,
            "suppress_long_dd":  -0.05,
            "suppress_short_rally": 0.05,
        },
    })),

    # --- 4. Long-only + drawdown filter (double bear protection) ---
    ("idea_3s_04_4h_long_only_dd", _merge3(_3STATE_BASE, {
        "strategy": {"long_only": True},
        "drawdown_filter": {
            "enabled":           True,
            "lookback_bars":     120,
            "suppress_long_dd":  -0.05,
            "suppress_short_rally": 0.05,
        },
    })),

    # --- 5. High confidence: only very clear 3-state regime calls ---
    ("idea_3s_05_4h_high_conf", _merge3(_3STATE_BASE, {
        "hmm": {
            "min_prob_hard":       0.70,
            "min_prob_hard_short": 0.70,
            "min_prob_entry":      0.72,
            "min_prob_exit":       0.50,
            "min_confidence":      0.55,
        },
    })),

    # --- 6. Low confidence: trade more bars (rely on entry filters) ---
    ("idea_3s_06_4h_low_conf", _merge3(_3STATE_BASE, {
        "hmm": {
            "min_prob_hard":       0.50,
            "min_prob_hard_short": 0.50,
            "min_prob_entry":      0.55,
            "min_prob_exit":       0.40,
            "min_confidence":      0.35,
        },
    })),

    # --- 7. Regime freshness filter: suppress stale regime entries ---
    ("idea_3s_07_4h_freshness", _merge3(_3STATE_BASE, {
        "regime": {
            "max_regime_freshness": 2.0,
            "freshness_decay_start": 1.0,
        },
    })),

    # --- 8. Trailing stop enabled (ride trending legs) ---
    ("idea_3s_08_4h_trailing", _merge3(_3STATE_BASE, {
        "strategies": {
            "trend": {
                "trailing_enabled":  True,
                "trailing_atr_mult": 1.5,
                "atr_tp_mult":       0.0,
            },
        },
    })),

    # --- 9. Higher regime persistence (only enter after 5 consecutive bars) ---
    ("idea_3s_09_4h_high_persist", _merge3(_3STATE_BASE, {
        "regime": {
            "persistence_bars":       5,
            "persistence_bars_short": 6,
        },
        "strategies": {
            "trend": {"persistence_bars": 5},
        },
    })),

    # --- 10. Range-only disabled, trend-only focus ---
    ("idea_3s_10_4h_trend_only", _merge3(_3STATE_BASE, {
        "strategies": {
            "range":    {"enabled": False},
            "volatile": {"enabled": False},
        },
    })),

    # --- 11. 1H regime TF instead of 4H (finer regime updates) ---
    ("idea_3s_11_1h_regime", _merge3(_3STATE_BASE, {
        "strategy": {"regime_timeframe": "1H"},
    })),

    # --- 12. 1H long-only with drawdown filter ---
    ("idea_3s_12_1h_long_only_dd", _merge3(_3STATE_BASE, {
        "strategy": {"regime_timeframe": "1H", "long_only": True},
        "drawdown_filter": {
            "enabled":           True,
            "lookback_bars":     120,
            "suppress_long_dd":  -0.05,
        },
    })),

    # --- 13. Self-transition gate: suppress unstable regimes (C1 filter) ---
    ("idea_3s_13_4h_self_tp_gate", _merge3(_3STATE_BASE, {
        "hmm": {
            "min_self_transition_prob": 0.85,
            "log_transition_matrix":    True,
        },
    })),

    # --- 14. BOS confirm: require LuxAlgo BOS/CHoCH for trend entries ---
    ("idea_3s_14_4h_bos_confirm", _merge3(_3STATE_BASE, {
        "strategies": {
            "trend": {
                "require_bos_confirm": True,
                "require_dc_breakout": False,
            },
        },
    })),

    # --- 15. Full combination: long-only + DD filter + freshness + trailing ---
    ("idea_3s_15_4h_full_combo", _merge3(_3STATE_BASE, {
        "strategy": {"long_only": True},
        "drawdown_filter": {
            "enabled":           True,
            "lookback_bars":     120,
            "suppress_long_dd":  -0.05,
        },
        "regime": {
            "max_regime_freshness": 2.0,
            "freshness_decay_start": 1.0,
        },
        "strategies": {
            "trend": {
                "trailing_enabled":  True,
                "trailing_atr_mult": 1.5,
                "atr_tp_mult":       0.0,
            },
        },
    })),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_pct(v):
    return "N/A" if v is None else f"{v*100:.1f}%"


def _safe_f(v, decimals=3):
    return "N/A" if v is None else f"{v:.{decimals}f}"


def _print_leaderboard(ranked):
    SEP  = "=" * 90
    DASH = "-" * 90
    print(f"\n{SEP}")
    print(f"  LEADERBOARD  (3-STATE HMM | filtered DD >= -25% | sorted by test return)")
    print(f"{SEP}")
    print(f"  {'#':<3} {'Experiment':<44} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'Verdict':<12}")
    print(f"{DASH}")
    for rank, (name, result) in enumerate(ranked, 1):
        tm     = result.get("test_metrics", {})
        ret    = _safe_pct(tm.get("annualized_return"))
        sharpe = _safe_f(tm.get("sharpe_ratio"))
        dd     = _safe_pct(tm.get("max_drawdown"))
        trades = tm.get("total_trades", "N/A")
        verdict = result.get("verdict", "N/A")
        print(f"  {rank:<3} {name:<44} {ret:>8} {sharpe:>7} {dd:>8} {str(trades):>7} {verdict:<12}")
    print(f"{SEP}\n")


def _print_wf_windows(wf):
    if not wf or not wf.get("windows"):
        return
    print(f"\n  WF Windows:")
    print(f"  {'Win':<4} {'Period':<35} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'Beats?':>7}")
    print("  " + "-" * 78)
    for w in wf["windows"]:
        beats = "YES" if w.get("beats_random_baseline") else "NO"
        print(f"  {w['window']:<4} {w.get('val_period', '?'):<35} "
              f"{w.get('annualized_return', 0) * 100:>7.1f}% "
              f"{w.get('sharpe_ratio', 0):>8.3f} "
              f"{w.get('total_trades', 0):>7} "
              f"{w.get('win_rate', 0) * 100:>7.1f}% "
              f"{beats:>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-state HMM scheme search: 15 ideas x 2 optimization targets = 30 experiments."
    )
    parser.add_argument("--config",    default="configs/base.yaml",
                        help="Base config path (default: configs/base.yaml)")
    parser.add_argument("--trials",    type=int, default=100,
                        help="Optuna trials per experiment (default: 100)")
    parser.add_argument("--top-wf",   type=int, default=5,
                        help="Re-run top N with walk-forward after main search (default: 5)")
    parser.add_argument("--dd-filter", type=float, default=-0.25,
                        help="Max drawdown filter for leaderboard (default: -0.25)")
    parser.add_argument("--ideas",    default=None,
                        help="Comma-separated 1-based idea indices to run (e.g. 1,3,5)")
    parser.add_argument("--no-wf",    action="store_true",
                        help="Skip walk-forward re-run of top results")
    args = parser.parse_args()

    # Load base config
    base_path   = PROJECT_ROOT / args.config
    base_config = load_config(base_path)

    # Results paths
    results_path    = PROJECT_ROOT / "artefacts" / "full_scheme_search_results_3state.json"
    main_results    = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
    wf_results_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_3state_wf_results.json"

    # Filter ideas if requested
    if args.ideas:
        indices = [int(i.strip()) - 1 for i in args.ideas.split(",")]
        ideas = [IDEA_OVERRIDES[i] for i in indices if 0 <= i < len(IDEA_OVERRIDES)]
    else:
        ideas = IDEA_OVERRIDES

    targets = ["val_sharpe", "val_return"]
    total = len(ideas) * len(targets)

    print(f"\n{'='*65}")
    print(f"  FULL SCHEME SEARCH -- 3-STATE HMM")
    print(f"  Ideas: {len(ideas)} | Targets: {len(targets)} | Experiments: {total}")
    print(f"  Optuna trials: {args.trials} | Top WF: {args.top_wf}")
    print(f"  Results -> {results_path.name}")
    print(f"{'='*65}\n")

    all_results = []
    start_time  = time.time()
    exp_num     = 0

    for idea_name, idea_override in ideas:
        for target in targets:
            exp_num += 1
            run_name = f"{idea_name}_opt_{target}"
            elapsed  = time.time() - start_time
            eta = (elapsed / exp_num) * (total - exp_num) if exp_num > 1 else 0

            print(f"\n{'#'*60}")
            print(f"  [{exp_num}/{total}] {run_name}")
            print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print(f"{'#'*60}")

            config = deepcopy(base_config)
            for k, v in idea_override.items():
                if isinstance(v, dict) and isinstance(config.get(k), dict):
                    config[k] = _merge3(config[k], v)
                else:
                    config[k] = deepcopy(v)

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
                result["config_override"] = idea_override

                tm = result.get("test_metrics", {})
                print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"dd={_safe_pct(tm.get('max_drawdown'))} "
                      f"trades={tm.get('total_trades', 'N/A')}"
                      f"  verdict={result.get('verdict')}")

            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                result = {
                    "run_name":        run_name,
                    "idea_name":       idea_name,
                    "optuna_target":   target,
                    "verdict":         "ERROR",
                    "test_metrics":    {"annualized_return": None, "sharpe_ratio": None, "max_drawdown": None},
                    "config_override": idea_override,
                    "error":           str(e),
                }

            all_results.append({"name": run_name, **result})

    # ---- Save results ----
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[3state_search] Results saved to {results_path}")

    # Merge into main results (for Streamlit dashboard)
    if main_results.exists():
        with open(main_results) as f:
            existing = json.load(f)
        new_names = {r["name"] for r in all_results}
        merged = [e for e in existing if e.get("name") not in new_names] + all_results
    else:
        merged = all_results
    with open(main_results, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"[3state_search] Merged into {main_results} ({len(merged)} total)")

    # ---- Rank ----
    passing = [
        (r["name"], r) for r in all_results
        if r.get("test_metrics", {}).get("max_drawdown") is not None
        and r["test_metrics"]["max_drawdown"] >= args.dd_filter
    ]
    ranked = sorted(
        passing,
        key=lambda x: x[1]["test_metrics"].get("annualized_return") or -999,
        reverse=True,
    )
    _print_leaderboard(ranked)

    # ---- Walk-forward on top N ----
    if ranked and args.top_wf > 0 and not args.no_wf:
        print(f"\n{'='*65}")
        print(f"  RE-RUNNING TOP {args.top_wf} WITH WALK-FORWARD VALIDATION")
        print(f"{'='*65}")
        wf_results = []

        for rank, (name, result) in enumerate(ranked[:args.top_wf], 1):
            print(f"\n[WF {rank}/{args.top_wf}] {name}")
            override = result.get("config_override", {})
            cfg = deepcopy(base_config)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = _merge3(cfg[k], v)
                else:
                    cfg[k] = deepcopy(v)

            try:
                wf_result = run_pipeline(
                    config         = cfg,
                    params         = result.get("params"),
                    retrain_model  = True,
                    walk_forward   = True,
                    optimize       = False,
                    legacy_signals = True,
                )
                wf_result["run_name"] = name + "_wf"
                wf_results.append((name, wf_result))

                tm = wf_result.get("test_metrics", {})
                wf = wf_result.get("walk_forward", {}) or {}
                print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                      f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                      f"wf_sharpe={_safe_f(wf.get('mean_sharpe'))} "
                      f"wf_pos={_safe_pct(wf.get('pct_positive'))} "
                      f"verdict={wf_result.get('verdict')}")
                _print_wf_windows(wf)

            except Exception as e:
                print(f"  !! WF FAILED: {e}")
                traceback.print_exc()

        if wf_results:
            wf_passing = [
                (n, r) for n, r in wf_results
                if r.get("test_metrics", {}).get("max_drawdown") is not None
                and r["test_metrics"]["max_drawdown"] >= args.dd_filter
            ]
            wf_ranked = sorted(
                wf_passing,
                key=lambda x: x[1]["test_metrics"].get("annualized_return") or -999,
                reverse=True,
            )
            print(f"\n{'='*65}")
            print(f"  FINAL WALK-FORWARD LEADERBOARD (3-STATE HMM)")
            _print_leaderboard(wf_ranked)

            with open(wf_results_path, "w") as f:
                json.dump([{"name": n, **r} for n, r in wf_results], f, indent=2, default=str)
            print(f"[3state_search] WF results saved to {wf_results_path}")

    # ---- Summary ----
    elapsed_total = time.time() - start_time
    print(f"\n[3state_search] Total time: {elapsed_total/60:.1f} minutes")
    print(f"[3state_search] Experiments: {len(all_results)} | "
          f"Passing DD filter: {len(passing)} | Ranked: {len(ranked)}")

    if ranked:
        best_name, best = ranked[0]
        tm = best.get("test_metrics", {})
        print(f"\n[3state_search] BEST RESULT: {best_name}")
        print(f"  Return  : {_safe_pct(tm.get('annualized_return'))}")
        print(f"  Sharpe  : {_safe_f(tm.get('sharpe_ratio'))}")
        print(f"  MaxDD   : {_safe_pct(tm.get('max_drawdown'))}")
        print(f"  Trades  : {tm.get('total_trades', 'N/A')}")
        print(f"  Verdict : {best.get('verdict')}")


if __name__ == "__main__":
    main()
