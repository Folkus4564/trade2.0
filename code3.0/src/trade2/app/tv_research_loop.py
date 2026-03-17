"""
app/tv_research_loop.py - TradingView Indicator Research Loop CLI.

Discovers TradingView indicators, translates Pine Script logic to Python via
Claude API, validates and integrates them as features, backtests, and iterates
until the goal is met or the idea budget is exhausted.

Usage:
    tv_research
    tv_research --max-ideas 5 --source seed --trials 50
    tv_research --dry-run --max-ideas 3
    tv_research --indicators supertrend,squeeze_momentum
    tv_research --goal-return 0.30 --goal-sharpe 1.2
"""

import argparse
import ast
import copy
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

PROJECT_ROOT = Path(__file__).parents[3]   # code3.0/
DATA_ROOT    = Path(__file__).parents[4]   # trade2.0/


# ---------------------------------------------------------------------------
# Config + seed helpers
# ---------------------------------------------------------------------------

def _load_seed_list(seed_path: Path) -> List[Dict[str, Any]]:
    import yaml
    if not seed_path.exists():
        return []
    with open(seed_path) as f:
        return yaml.safe_load(f).get("indicators", [])


def _load_log(log_path: Path) -> List[Dict[str, Any]]:
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def _save_log(log_path: Path, log: List[Dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)


def _save_best(best_path: Path, entry: Dict[str, Any]) -> None:
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w") as f:
        json.dump(entry, f, indent=2, default=str)


def _get_tried_names(log: List[Dict[str, Any]]) -> set:
    return {entry["name"] for entry in log}


# ---------------------------------------------------------------------------
# Indicator selection
# ---------------------------------------------------------------------------

def _pick_from_seed(
    seed_list: List[Dict],
    tried: set,
    force_indicators: Optional[List[str]] = None,
) -> Optional[Dict]:
    if force_indicators:
        for ind in seed_list:
            if ind["name"] in force_indicators and ind["name"] not in tried:
                return ind
        # Return a placeholder dict for forced indicators not in seed list
        for name in force_indicators:
            if name not in tried:
                return {
                    "name": name,
                    "category": "unknown",
                    "description": f"{name} indicator",
                    "default_params": {},
                    "integration_mode": "feature",
                }
        return None
    for ind in seed_list:
        if ind["name"] not in tried:
            return ind
    return None


def _discover_from_claude(
    client: anthropic.Anthropic,
    tried: set,
    model: str,
) -> Optional[Dict]:
    """Ask Claude to suggest a new TradingView indicator not already tried."""
    tried_str = ", ".join(sorted(tried)) if tried else "none"
    with client.messages.stream(
        model=model,
        max_tokens=512,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": (
                "Suggest one well-known TradingView community indicator that:\n"
                "1. Has clear mathematical rules (not AI-based)\n"
                "2. Can be computed from OHLCV price data\n"
                f"3. Has NOT been tried yet: [{tried_str}]\n\n"
                "Respond with ONLY a JSON object:\n"
                '{\n'
                '  "name": "snake_case_name",\n'
                '  "category": "trend|momentum|volatility|volume|oscillator",\n'
                '  "description": "one sentence description",\n'
                '  "default_params": {"key": value},\n'
                '  "integration_mode": "feature"\n'
                '}'
            ),
        }],
    ) as stream:
        response = stream.get_final_message()

    text = next((b.text for b in response.content if b.type == "text"), "")
    json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Code translation + validation
# ---------------------------------------------------------------------------

def _translate_to_python(
    client: anthropic.Anthropic,
    indicator: Dict[str, Any],
    cdc_source: str,
    model: str,
) -> Optional[str]:
    """Translate indicator description to Python using Claude."""
    name     = indicator["name"]
    category = indicator.get("category", "unknown")
    desc     = indicator.get("description", "")
    params   = indicator.get("default_params", {})

    system_prompt = (
        "You translate TradingView indicator logic to Python following this exact pattern:\n\n"
        f"{cdc_source}\n\n"
        "Rules:\n"
        f"1. Function name MUST be: add_{name}_features(df, config) -> pd.DataFrame\n"
        f"2. Read params from config.get('tv_indicators', {{}}).get('{name}', {{}})\n"
        "3. All output columns shift(1) for lag safety\n"
        "4. Use talib where possible, otherwise numpy/pandas\n"
        f"5. Return df copy with new columns prefixed: {name}_\n"
        "6. No lookahead bias (never use future data)\n"
        "7. Include all imports at top of the code (numpy, pandas, talib)\n"
        "8. No class definitions, no global state\n"
        "9. Output ONLY the Python code, no explanation, no markdown code blocks"
    )

    user_msg = (
        f"Translate this indicator to Python:\n"
        f"Name: {name}\n"
        f"Category: {category}\n"
        f"Description: {desc}\n"
        f"Default params: {json.dumps(params)}\n\n"
        f"Write the add_{name}_features(df, config) function."
    )

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        response = stream.get_final_message()

    code = next((b.text for b in response.content if b.type == "text"), "")
    # Strip markdown code blocks if present
    code = re.sub(r'^```python\s*', '', code.strip(), flags=re.MULTILINE)
    code = re.sub(r'```\s*$', '', code.strip(), flags=re.MULTILINE)
    return code.strip() or None


def _validate_code(code: str, name: str) -> Tuple[bool, str]:
    """Validate Python code: parse + basic structure check."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    if f"add_{name}_features" not in code:
        return False, f"Missing function add_{name}_features"
    return True, "syntax ok"


def _test_run(code: str, name: str) -> Tuple[bool, str]:
    """Test-run the indicator via exec on a small dummy DataFrame."""
    import numpy as np
    import pandas as pd

    try:
        # Build exec namespace with common libraries
        namespace: Dict[str, Any] = {
            "np": np,
            "pd": pd,
            "__name__": f"__tv_test_{name}__",
        }
        try:
            import talib
            namespace["talib"] = talib
        except ImportError:
            pass

        exec(compile(code, f"<tv_{name}>", "exec"), namespace)  # noqa: S102

        add_fn = namespace.get(f"add_{name}_features")
        if add_fn is None:
            return False, f"add_{name}_features not found after exec"

        # Dummy OHLCV (200 bars)
        np.random.seed(42)
        n = 200
        close = 1800.0 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "Open":   close - np.abs(np.random.randn(n)) * 2,
            "High":   close + np.abs(np.random.randn(n)) * 3,
            "Low":    close - np.abs(np.random.randn(n)) * 3,
            "Close":  close,
            "Volume": np.abs(np.random.randn(n)) * 1000 + 5000,
        })

        result = add_fn(df.copy(), {})

        if not isinstance(result, pd.DataFrame):
            return False, "Function must return a DataFrame"
        new_cols = [c for c in result.columns if c not in df.columns]
        if not new_cols:
            return False, "No new columns added to DataFrame"
        return True, f"OK: added {len(new_cols)} columns ({', '.join(new_cols[:3])})"

    except Exception as e:
        return False, f"RuntimeError: {e}"


def _write_module(code: str, name: str, tv_indicators_dir: Path) -> Path:
    """Write indicator module to permanent location in the package."""
    module_path = tv_indicators_dir / f"{name}.py"
    # Add docstring header
    header = (
        f'"""\n'
        f'features/tv_indicators/{name}.py - Auto-generated TV indicator module.\n'
        f'Generated by tv_research_loop on {datetime.now().isoformat()[:19]}\n'
        f'"""\n'
    )
    # Only prepend header if code doesn't start with a docstring
    if code.startswith('"""') or code.startswith("'''"):
        full_code = code
    else:
        full_code = header + code

    with open(module_path, "w") as f:
        f.write(full_code)
    return module_path


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------

def _build_run_config(base_config: Dict, indicator: Dict) -> Dict:
    """Return a deep-copy of base_config with the TV indicator enabled."""
    cfg  = copy.deepcopy(base_config)
    name = indicator["name"]
    if "tv_indicators" not in cfg:
        cfg["tv_indicators"] = {}
    cfg["tv_indicators"][name] = {
        "enabled": True,
        **indicator.get("default_params", {}),
    }
    return cfg


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    config: Dict,
    n_trials: int,
    walk_forward: bool,
    retrain_model: bool,
) -> Dict:
    from trade2.app.run_pipeline import run_pipeline
    return run_pipeline(
        config          = config,
        walk_forward    = walk_forward,
        retrain_model   = retrain_model,
        export_approved = False,
        optimize        = n_trials > 0,
        n_trials        = n_trials,
    )


# ---------------------------------------------------------------------------
# Goal check
# ---------------------------------------------------------------------------

def _goal_met(metrics: Dict, tv_cfg: Dict) -> bool:
    goal_return = tv_cfg.get("goal_return", 0.50)
    goal_sharpe = tv_cfg.get("goal_sharpe", 1.5)
    goal_max_dd = tv_cfg.get("goal_max_dd", -0.25)
    return (
        metrics.get("annualized_return", 0) >= goal_return
        and metrics.get("sharpe_ratio", 0)   >= goal_sharpe
        and metrics.get("max_drawdown", -999) >= goal_max_dd
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _print_leaderboard(log: List[Dict]) -> None:
    completed = [e for e in log if e.get("status") == "COMPLETED" and e.get("test_metrics")]
    if not completed:
        print("\n[tv_research] No completed runs to rank.")
        return
    ranked = sorted(completed, key=lambda x: x["test_metrics"].get("annualized_return", 0), reverse=True)
    sep  = "=" * 76
    dash = "-" * 76
    print(f"\n{sep}")
    print("  TV INDICATOR RESEARCH - LEADERBOARD")
    print(sep)
    print(f"  {'#':<3} {'Indicator':<25} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Verdict':<14}")
    print(dash)
    for i, entry in enumerate(ranked[:10], 1):
        tm = entry.get("test_metrics", {})
        best_marker = " *" if entry.get("is_best") else ""
        print(
            f"  {i:<3} {entry['name']:<25}"
            f" {tm.get('annualized_return', 0)*100:>7.1f}%"
            f" {tm.get('sharpe_ratio', 0):>8.3f}"
            f" {tm.get('max_drawdown', 0)*100:>7.1f}%"
            f"  {entry.get('verdict', '?'):<14}{best_marker}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView Indicator Research Loop")
    parser.add_argument("--max-ideas",     type=int,   default=30)
    parser.add_argument("--source",        default="both", choices=["seed", "web", "both"])
    parser.add_argument("--trials",        type=int,   default=100)
    parser.add_argument("--model",         default=None)
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--indicators",    default=None, help="Comma-separated specific indicators")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--goal-return",   type=float, default=None)
    parser.add_argument("--goal-sharpe",   type=float, default=None)
    parser.add_argument("--walk-forward",  action="store_true")
    parser.add_argument("--retrain-model", action="store_true")
    parser.add_argument("--base-config",   default="configs/base.yaml")
    args = parser.parse_args()

    # ---- Paths ----
    base_cfg_path    = PROJECT_ROOT / args.base_config
    tv_research_dir  = PROJECT_ROOT / "artefacts" / "tv_research"
    tv_indicators_dir = PROJECT_ROOT / "src" / "trade2" / "features" / "tv_indicators"
    tv_research_dir.mkdir(parents=True, exist_ok=True)
    tv_indicators_dir.mkdir(parents=True, exist_ok=True)
    log_path  = tv_research_dir / "tv_research_log.json"
    best_path = tv_research_dir / "tv_research_best.json"
    pine_dir  = tv_research_dir / "pine_scripts"
    pine_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load base config ----
    from trade2.config.loader import load_config
    base_config = load_config(str(base_cfg_path))
    tv_cfg      = base_config.get("tv_research", {})

    model     = args.model or tv_cfg.get("model", "claude-sonnet-4-6")
    max_ideas = args.max_ideas
    n_trials  = args.trials

    if args.goal_return is not None:
        tv_cfg["goal_return"] = args.goal_return
    if args.goal_sharpe is not None:
        tv_cfg["goal_sharpe"] = args.goal_sharpe

    # ---- Load seed list ----
    seed_path = PROJECT_ROOT / tv_cfg.get("seed_list", "configs/tv_seed_list.yaml")
    seed_list = _load_seed_list(seed_path)

    # ---- Existing log ----
    log   = _load_log(log_path)
    tried = _get_tried_names(log) if args.skip_existing else set()

    force_indicators = (
        [x.strip() for x in args.indicators.split(",")]
        if args.indicators else None
    )

    # ---- CDC template for translation prompt ----
    cdc_path   = PROJECT_ROOT / "src" / "trade2" / "features" / "cdc.py"
    cdc_source = cdc_path.read_text() if cdc_path.exists() else ""

    # ---- Baseline metrics ----
    verdict_path    = PROJECT_ROOT / "artefacts" / "reports" / "final_verdict.json"
    baseline_metrics: Dict = {}
    if verdict_path.exists():
        with open(verdict_path) as f:
            baseline_metrics = json.load(f).get("test_metrics", {})
        print(
            f"[tv] Baseline: return={baseline_metrics.get('annualized_return', 0)*100:.1f}%"
            f" | sharpe={baseline_metrics.get('sharpe_ratio', 0):.3f}"
        )

    # ---- Anthropic client ----
    client = anthropic.Anthropic()

    best_return = baseline_metrics.get("annualized_return", 0.0)
    best_entry: Optional[Dict] = None

    print(f"\n[tv] TradingView Indicator Research Loop")
    print(f"[tv] max_ideas={max_ideas} | model={model} | trials={n_trials} | dry_run={args.dry_run}")
    print(f"[tv] source={args.source} | seed_count={len(seed_list)}")
    print(f"[tv] goals: return>={tv_cfg.get('goal_return', 0.50)*100:.0f}%"
          f" | sharpe>={tv_cfg.get('goal_sharpe', 1.5):.1f}"
          f" | dd>={tv_cfg.get('goal_max_dd', -0.25)*100:.0f}%\n")

    for iteration in range(max_ideas):
        print(f"\n[tv] === Iteration {iteration+1}/{max_ideas} ===")

        # ---- Pick indicator ----
        indicator: Optional[Dict] = None
        source = "seed"

        if args.source in ("seed", "both"):
            indicator = _pick_from_seed(seed_list, tried, force_indicators)
            source    = "seed"

        if indicator is None and args.source in ("web", "both"):
            print("[tv] Seed exhausted — asking Claude for new indicator...")
            try:
                indicator = _discover_from_claude(client, tried, model)
                source    = "discovery"
            except Exception as e:
                print(f"[tv] Discovery error: {e}")

        if indicator is None:
            print("[tv] No more indicators available. Stopping.")
            break

        name = indicator["name"]
        tried.add(name)
        print(f"[tv] Indicator: {name} ({indicator.get('category', '?')}) [{source}]")
        print(f"[tv]   {indicator.get('description', '')}")

        entry: Dict[str, Any] = {
            "id":                len(log) + 1,
            "name":              name,
            "category":          indicator.get("category", "unknown"),
            "source":            source,
            "integration_mode":  indicator.get("integration_mode", "feature"),
            "status":            "STARTED",
            "timestamp":         datetime.now().isoformat()[:19],
            "duration_seconds":  0,
            "python_module_path": None,
            "config_override":   None,
            "test_metrics":      None,
            "baseline_metrics":  baseline_metrics,
            "delta_vs_baseline": None,
            "verdict":           None,
            "is_best":           False,
            "error":             None,
        }
        t0 = time.time()

        # ---- Translate ----
        print(f"[tv] Translating {name} to Python via Claude ({model})...")
        try:
            code = _translate_to_python(client, indicator, cdc_source, model)
            if not code:
                raise ValueError("Translation returned empty output")
        except Exception as e:
            entry.update(status="TRANSLATION_FAILED", error=str(e),
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[tv] TRANSLATION FAILED: {e}")
            continue

        # ---- Validate syntax ----
        valid, msg = _validate_code(code, name)
        if not valid:
            entry.update(status="CODE_ERROR", error=f"Validation: {msg}",
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[tv] CODE ERROR: {msg}")
            continue
        print(f"[tv] Validation: {msg}")

        # ---- Test run ----
        print(f"[tv] Test-running {name}...")
        ok, run_msg = _test_run(code, name)
        if not ok:
            entry.update(status="CODE_ERROR", error=f"TestRun: {run_msg}",
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[tv] TEST RUN FAILED: {run_msg}")
            continue
        print(f"[tv] Test run: {run_msg}")

        # ---- Write module ----
        module_path = _write_module(code, name, tv_indicators_dir)
        entry["python_module_path"] = str(module_path.relative_to(PROJECT_ROOT))
        print(f"[tv] Module written: {module_path.name}")

        if args.dry_run:
            entry.update(status="DRY_RUN", duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print("[tv] DRY RUN: pipeline skipped")
            continue

        # ---- Build config override ----
        run_config = _build_run_config(base_config, indicator)
        entry["config_override"] = {"tv_indicators": {name: run_config["tv_indicators"][name]}}

        # ---- Run pipeline ----
        print(f"[tv] Running pipeline (optimize={n_trials > 0}, trials={n_trials})...")
        try:
            results     = _run_pipeline(run_config, n_trials, args.walk_forward, args.retrain_model)
            test_metrics = results.get("test_metrics", {})
            verdict      = results.get("verdict", "?")

            entry["status"]      = "COMPLETED"
            entry["test_metrics"] = test_metrics
            entry["verdict"]      = verdict

            if baseline_metrics:
                entry["delta_vs_baseline"] = {
                    "return_delta": (test_metrics.get("annualized_return", 0)
                                     - baseline_metrics.get("annualized_return", 0)),
                    "sharpe_delta": (test_metrics.get("sharpe_ratio", 0)
                                     - baseline_metrics.get("sharpe_ratio", 0)),
                    "dd_delta":     (test_metrics.get("max_drawdown", 0)
                                     - baseline_metrics.get("max_drawdown", 0)),
                }

            cur_return = test_metrics.get("annualized_return", 0)
            if cur_return > best_return:
                best_return = cur_return
                entry["is_best"] = True
                best_entry = copy.deepcopy(entry)
                _save_best(best_path, best_entry)
                print(f"[tv] NEW BEST: {name} | return={cur_return*100:.1f}%")

            print(
                f"[tv] Done: return={cur_return*100:.1f}%"
                f" | sharpe={test_metrics.get('sharpe_ratio', 0):.3f}"
                f" | verdict={verdict}"
            )

            if _goal_met(test_metrics, tv_cfg):
                print(f"\n[tv] GOAL MET by {name}! Stopping loop.")
                entry["duration_seconds"] = int(time.time() - t0)
                log.append(entry)
                _save_log(log_path, log)
                break

        except Exception as e:
            entry.update(
                status="PIPELINE_FAILED",
                error=traceback.format_exc(),
                verdict="ERROR",
            )
            print(f"[tv] PIPELINE FAILED: {e}")

        entry["duration_seconds"] = int(time.time() - t0)
        log.append(entry)
        _save_log(log_path, log)

    # ---- Final summary ----
    _print_leaderboard(log)
    if best_entry:
        tm = best_entry.get("test_metrics", {})
        print(f"\n[tv] Best: {best_entry['name']}"
              f" | return={tm.get('annualized_return', 0)*100:.1f}%"
              f" | sharpe={tm.get('sharpe_ratio', 0):.3f}")
    print(f"\n[tv] Log: {log_path}")
    print(f"[tv] Best: {best_path}")


if __name__ == "__main__":
    main()
