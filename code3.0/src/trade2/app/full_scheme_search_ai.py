"""
app/full_scheme_search_ai.py - AI-driven scheme search.

Claude reads previous results + config schema, proposes new config overrides,
then runs them through the pipeline automatically.

Usage:
    full_scheme_search_ai --rounds 3 --ideas-per-round 5 --trials 100 --top-wf 3
    full_scheme_search_ai --dry-run --ideas-per-round 3   # test prompt only
"""

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

from trade2.config.loader import load_config
from trade2.app.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).parents[3]  # code3.0/

# ---------------------------------------------------------------------------
# Helpers (same as v3)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Schema + results helpers for prompt construction
# ---------------------------------------------------------------------------

def _build_schema_summary(cfg: dict, indent: int = 0) -> str:
    """Build a readable section -> key -> value table from a nested config dict."""
    lines = []
    pad = "  " * indent
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"{pad}[{k}]")
            lines.append(_build_schema_summary(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{k}: {v}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def _summarize_override(override: dict, prefix: str = "") -> str:
    """Flatten a nested override dict to a short string for display."""
    parts = []
    for k, v in override.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            parts.append(_summarize_override(v, key))
        else:
            parts.append(f"{key}={v}")
    return " ".join(parts)


def _build_results_summary(existing: list) -> str:
    """Build a compact table of previous experiment results for the prompt."""
    if not existing:
        return "  (no previous experiments yet)"

    header = (
        f"  {'Name':<44} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} "
        f"{'Trades':>7} {'Verdict':<15} Key overrides"
    )
    rows = [header, "  " + "-" * 115]

    # Show last 60 experiments to stay within reasonable prompt size
    for r in existing[-60:]:
        name    = str(r.get("name", r.get("run_name", "?")))[:43]
        tm      = r.get("test_metrics", {})
        ret     = _safe_pct(tm.get("annualized_return"))
        sharpe  = _safe_f(tm.get("sharpe_ratio"))
        dd      = _safe_pct(tm.get("max_drawdown"))
        trades  = str(tm.get("total_trades", "N/A"))
        verdict = str(r.get("verdict", "N/A"))[:14]
        override_str = _summarize_override(r.get("config_override", {}))[:60]
        rows.append(
            f"  {name:<44} {ret:>8} {sharpe:>7} {dd:>8} "
            f"{trades:>7} {verdict:<15} {override_str}"
        )
    return "\n".join(rows)


def _validate_override(override: dict, base: dict, path: str = "") -> dict:
    """
    Recursively strip keys not present in base config.
    Warns for each unknown key but does not crash.
    """
    cleaned = {}
    for k, v in override.items():
        full_path = f"{path}{k}"
        if k not in base:
            print(f"  [AI] WARNING: unknown config key '{full_path}' stripped")
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            nested = _validate_override(v, base[k], f"{full_path}.")
            if nested:
                cleaned[k] = nested
        else:
            cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Core AI idea generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a quant research assistant specialising in XAUUSD systematic trading. "
    "Your job is to propose new config override experiments based on previous results. "
    "Be analytical: build on what worked, avoid repeating what failed."
)


def _build_user_prompt(schema_summary: str, results_summary: str, n_ideas: int) -> str:
    return f"""You are driving an automated XAUUSD strategy search.

## Config Schema  (section -> key -> current_default)
{schema_summary}

## Previous Experiment Results
{results_summary}

## Performance Targets
- Annualized Return : target >= 50%, minimum >= 20%
- Sharpe Ratio      : target >= 1.5,  minimum >= 1.0
- Max Drawdown      : target <= -20%, minimum <= -35%
- Profit Factor     : target >= 1.5,  minimum >= 1.2
- Total Trades      : minimum 30

## Your Task
Propose exactly {n_ideas} NEW config override experiments.

Rules:
1. Only use keys that appear in the config schema above - never invent new keys
2. Each experiment must have a unique descriptive name in snake_case
3. Build on combinations that performed well; avoid repeating known failures
4. Vary strategies: regime params, session filters, sub-strategy toggles, trailing stops, sizing, confidence thresholds
5. Respond ONLY with a valid JSON array - no markdown fences, no prose outside the array

Required JSON format:
[
  {{
    "name": "idea_ai_short_description",
    "override": {{
      "section_name": {{"key": value}}
    }},
    "rationale": "1-2 sentence explanation of why this might improve performance"
  }}
]
"""


def generate_ideas(
    client,
    base_config: dict,
    existing_results: list,
    n_ideas: int,
    model: str,
) -> list:
    """
    Call the Anthropic API to generate n_ideas new experiment overrides.

    Returns list of (name, override_dict, rationale) tuples.
    Strips unknown config keys with a warning.
    """
    schema_summary  = _build_schema_summary(base_config)
    results_summary = _build_results_summary(existing_results)
    user_prompt     = _build_user_prompt(schema_summary, results_summary, n_ideas)

    print(f"  [AI] Calling {model} for {n_ideas} ideas ...")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )

    # Strip optional markdown code fences
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first fence line and last fence line
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    proposals = json.loads(text)  # raises if malformed - caller handles

    ideas = []
    for p in proposals:
        name      = str(p.get("name", f"idea_ai_{int(time.time())}"))
        raw_ov    = p.get("override", {})
        rationale = str(p.get("rationale", ""))

        # Validate and strip unknown keys
        override = _validate_override(raw_ov, base_config)

        ideas.append((name, override, rationale))
        print(f"  [AI] Proposed: {name}")
        if rationale:
            print(f"       Rationale: {rationale}")

    return ideas


# ---------------------------------------------------------------------------
# Shared finish: rank, save, merge, WF
# ---------------------------------------------------------------------------

def _finish(all_round_results, base_config, args, main_results, ai_results_path, ai_search_dir):
    if not all_round_results:
        print(f"\n[ai_search] No results collected. Exiting.")
        return

    dd_filter = args.dd_filter
    passing = [
        (r["name"], r) for r in all_round_results
        if r.get("test_metrics", {}).get("max_drawdown") is not None
        and r["test_metrics"]["max_drawdown"] >= dd_filter
    ]
    ranked = sorted(
        passing,
        key=lambda x: x[1]["test_metrics"].get("annualized_return") or -999,
        reverse=True,
    )

    _print_leaderboard(ranked)

    # Save AI results file
    with open(ai_results_path, "w") as f:
        json.dump(all_round_results, f, indent=2, default=str)
    print(f"[ai_search] AI results saved to {ai_results_path}")

    # Merge into main results
    if not args.no_merge:
        existing_main = _load_existing(main_results)
        ai_names = {r.get("idea_name", "") for r in all_round_results}
        existing_main = [e for e in existing_main if e.get("idea_name") not in ai_names]
        merged = existing_main + all_round_results
        with open(main_results, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"[ai_search] Merged into {main_results} ({len(merged)} total experiments)")

    # Walk-forward re-run of top N
    if ranked and args.top_wf > 0:
        print(f"\n{'='*65}")
        print(f"  RE-RUNNING TOP {args.top_wf} WITH WALK-FORWARD VALIDATION")
        print(f"{'='*65}")
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
            print(f"\n{'='*65}")
            print(f"  FINAL WALK-FORWARD LEADERBOARD")
            _print_leaderboard(wf_ranked)

            wf_path = ai_search_dir.parent / "full_scheme_search_ai_wf_results.json"
            with open(wf_path, "w") as f:
                json.dump([{"name": n, **r} for n, r in wf_results], f, indent=2, default=str)
            print(f"[ai_search] WF results saved to {wf_path}")

    # Summary
    print(f"\n[ai_search] Experiments: {len(all_round_results)} | "
          f"Passing DD filter: {len(passing)} | Ranked: {len(ranked)}")

    if ranked:
        best_name, best_result = ranked[0]
        best_tm = best_result.get("test_metrics", {})
        print(f"\n[ai_search] BEST RESULT: {best_name}")
        print(f"  Return  : {_safe_pct(best_tm.get('annualized_return'))}")
        print(f"  Sharpe  : {_safe_f(best_tm.get('sharpe_ratio'))}")
        print(f"  MaxDD   : {_safe_pct(best_tm.get('max_drawdown'))}")
        print(f"  Trades  : {best_tm.get('total_trades', 'N/A')}")
        print(f"  Verdict : {best_result.get('verdict')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-driven scheme search: Claude proposes overrides, pipeline runs them."
    )
    parser.add_argument("--config",          default="configs/xauusd_mtf.yaml")
    parser.add_argument("--base-config",     default="configs/base.yaml")
    parser.add_argument("--rounds",          type=int,   default=3)
    parser.add_argument("--ideas-per-round", type=int,   default=5)
    parser.add_argument("--trials",          type=int,   default=100)
    parser.add_argument("--top-wf",          type=int,   default=3)
    parser.add_argument("--dd-filter",       type=float, default=-0.25)
    parser.add_argument("--model",           default="claude-sonnet-4-6")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Generate and print ideas without running the pipeline")
    parser.add_argument("--ideas-file",      default=None,
                        help="Path to a pre-generated ideas JSON file (skips API call)")
    parser.add_argument("--no-merge",        action="store_true",
                        help="Do not merge results into full_scheme_search_results.json")
    args = parser.parse_args()

    # Load base config
    base_path     = PROJECT_ROOT / args.base_config
    override_path = PROJECT_ROOT / args.config if args.config != args.base_config else None
    base_config   = load_config(base_path, override_path)

    # Paths
    ai_search_dir  = PROJECT_ROOT / "artefacts" / "ai_search"
    ai_search_dir.mkdir(parents=True, exist_ok=True)
    main_results    = PROJECT_ROOT / "artefacts" / "full_scheme_search_results.json"
    ai_results_path = PROJECT_ROOT / "artefacts" / "full_scheme_search_results_ai.json"

    # --ideas-file mode: load pre-generated ideas, skip API entirely
    if args.ideas_file:
        ideas_path = Path(args.ideas_file)
        if not ideas_path.is_absolute():
            ideas_path = PROJECT_ROOT / args.ideas_file
        with open(ideas_path) as f:
            raw = json.load(f)
        preloaded_ideas = [
            (p["name"], p["override"], p.get("rationale", ""))
            for p in raw
        ]
        print(f"\n{'='*65}")
        print(f"  FULL SCHEME SEARCH  --  AI-DRIVEN (ideas-file mode)")
        print(f"  Ideas loaded from: {ideas_path.name}")
        print(f"  Ideas: {len(preloaded_ideas)} | Targets: 2 | Experiments: {len(preloaded_ideas)*2}")
        print(f"  Optuna trials: {args.trials} | Top WF: {args.top_wf}")
        print(f"{'='*65}\n")

        all_round_results: list = []
        targets    = ["val_sharpe", "val_return"]
        start_time = time.time()
        total      = len(preloaded_ideas) * len(targets)
        exp_num    = 0

        existing_names = {
            r.get("idea_name", r.get("name", ""))
            for r in _load_existing(main_results)
        }
        unique_ideas = []
        for name, override, rationale in preloaded_ideas:
            if name in existing_names:
                print(f"  [skip] duplicate: {name}")
            else:
                unique_ideas.append((name, override, rationale))
                existing_names.add(name)

        for idea_name, override, rationale in unique_ideas:
            for target in targets:
                exp_num += 1
                run_name = f"{idea_name}_opt_{target}"
                elapsed  = time.time() - start_time
                eta = (elapsed / exp_num) * (total - exp_num) if exp_num > 1 else 0

                print(f"\n{'#'*60}")
                print(f"  [{exp_num}/{total}] {run_name}")
                print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                if rationale:
                    print(f"  Rationale: {rationale}")
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
                    result["ai_rationale"]    = rationale
                    all_round_results.append({"name": run_name, **result})

                    tm = result.get("test_metrics", {})
                    print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                          f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                          f"dd={_safe_pct(tm.get('max_drawdown'))} "
                          f"trades={tm.get('total_trades','N/A')}")

                except Exception as e:
                    print(f"  !! FAILED: {e}")
                    traceback.print_exc()
                    all_round_results.append({
                        "name": run_name, "run_name": run_name,
                        "idea_name": idea_name, "optuna_target": target,
                        "verdict": "ERROR",
                        "test_metrics": {"annualized_return": None, "sharpe_ratio": None, "max_drawdown": None},
                        "config_override": override, "ai_rationale": rationale, "error": str(e),
                    })

        # --- filter/rank/save/wf (shared with main loop below) ---
        _finish(all_round_results, base_config, args, main_results, ai_results_path, ai_search_dir)
        sys.exit(0)

    # ------------------------------------------------------------------
    # API mode: requires ANTHROPIC_API_KEY
    # ------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ai_search] ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Set it with: set ANTHROPIC_API_KEY=sk-ant-...")
        print("  Or use --ideas-file to load pre-generated ideas without the API.")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("[ai_search] ERROR: anthropic package not installed.")
        print("  Install with: pip install -e \"code3.0/[ai]\"")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"\n{'='*65}")
    print(f"  FULL SCHEME SEARCH  --  AI-DRIVEN")
    print(f"  Rounds: {args.rounds} | Ideas/round: {args.ideas_per_round}")
    print(f"  Targets per idea: 2 (val_sharpe + val_return)")
    print(f"  Total max experiments: {args.rounds * args.ideas_per_round * 2}")
    print(f"  Optuna trials per experiment: {args.trials}")
    print(f"  Model: {args.model}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"{'='*65}\n")

    # Accumulated results across all rounds
    all_round_results: list = []
    targets = ["val_sharpe", "val_return"]
    start_time = time.time()

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'#'*65}")
        print(f"  ROUND {round_num}/{args.rounds}")
        print(f"{'#'*65}")

        # Load all existing results so AI has full context
        existing = _load_existing(main_results) + all_round_results

        # Build set of already-run names for deduplication
        existing_names = {
            r.get("idea_name", r.get("name", "")) for r in existing
        }

        # Generate ideas
        try:
            ideas = generate_ideas(
                client, base_config, existing, args.ideas_per_round, args.model
            )
        except Exception as e:
            print(f"  [AI] Round {round_num} idea generation FAILED: {e}")
            traceback.print_exc()
            print(f"  [AI] Skipping round {round_num}.")
            continue

        # Deduplicate
        unique_ideas = []
        for name, override, rationale in ideas:
            if name in existing_names:
                print(f"  [AI] Skipping duplicate: {name}")
            else:
                unique_ideas.append((name, override, rationale))
                existing_names.add(name)

        if not unique_ideas:
            print(f"  [AI] No unique ideas for round {round_num}, skipping.")
            continue

        # Save raw ideas to audit trail
        ideas_audit = [
            {"name": n, "override": ov, "rationale": r}
            for n, ov, r in unique_ideas
        ]
        audit_path = ai_search_dir / f"round_{round_num}_ideas.json"
        with open(audit_path, "w") as f:
            json.dump(ideas_audit, f, indent=2, default=str)
        print(f"\n  [AI] Ideas saved to {audit_path}")

        if args.dry_run:
            print(f"\n  [DRY-RUN] Ideas for round {round_num}:")
            for i, (name, override, rationale) in enumerate(unique_ideas, 1):
                print(f"    {i}. {name}")
                print(f"       override: {json.dumps(override)}")
                print(f"       rationale: {rationale}")
            continue

        # Run experiments
        total_this_round = len(unique_ideas) * len(targets)
        exp_num = 0

        for idea_name, override, rationale in unique_ideas:
            for target in targets:
                exp_num += 1
                run_name = f"{idea_name}_opt_{target}"
                elapsed  = time.time() - start_time
                eta = (elapsed / exp_num) * (total_this_round - exp_num) if exp_num > 1 else 0

                print(f"\n{'#'*60}")
                print(f"  [R{round_num} {exp_num}/{total_this_round}] {run_name}")
                print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                print(f"  Rationale: {rationale}")
                print(f"{'#'*60}")

                config = _deep_merge(base_config, override)

                try:
                    result = run_pipeline(
                        config        = config,
                        retrain_model = True,
                        walk_forward  = False,
                        optimize      = True,
                        n_trials      = args.trials,
                        legacy_signals = True,
                        optuna_target = target,
                    )
                    result["run_name"]        = run_name
                    result["idea_name"]       = idea_name
                    result["optuna_target"]   = target
                    result["config_override"] = override
                    result["ai_rationale"]    = rationale
                    result["ai_round"]        = round_num
                    all_round_results.append({"name": run_name, **result})

                    tm = result.get("test_metrics", {})
                    print(f"  >> return={_safe_pct(tm.get('annualized_return'))} "
                          f"sharpe={_safe_f(tm.get('sharpe_ratio'))} "
                          f"dd={_safe_pct(tm.get('max_drawdown'))} "
                          f"trades={tm.get('total_trades','N/A')}")

                except Exception as e:
                    print(f"  !! FAILED: {e}")
                    traceback.print_exc()
                    all_round_results.append({
                        "name":           run_name,
                        "run_name":       run_name,
                        "idea_name":      idea_name,
                        "optuna_target":  target,
                        "verdict":        "ERROR",
                        "test_metrics":   {
                            "annualized_return": None,
                            "sharpe_ratio":      None,
                            "max_drawdown":      None,
                        },
                        "config_override": override,
                        "ai_rationale":    rationale,
                        "ai_round":        round_num,
                        "error":           str(e),
                    })

    if args.dry_run:
        print(f"\n[ai_search] Dry-run complete. No experiments were run.")
        sys.exit(0)

    total_elapsed = time.time() - start_time
    print(f"\n[ai_search] Total time: {total_elapsed/60:.1f} minutes")
    _finish(all_round_results, base_config, args, main_results, ai_results_path, ai_search_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
