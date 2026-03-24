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
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parents[3]   # code3.0/
DATA_ROOT    = Path(__file__).parents[4]   # trade2.0/

# Load .env from code3.0/ if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

_DEFAULT_MODEL = {
    "claude":   "claude-sonnet-4-6",
    "deepseek": "deepseek-reasoner",
}


def _build_client(provider: str):
    if provider == "claude":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "deepseek":
        import openai
        return openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _llm_complete(
    client,
    provider: str,
    model: str,
    user_msg: str,
    system: str = "",
    max_tokens: int = 2048,
) -> str:
    """Unified LLM call: returns response text for Claude or DeepSeek."""
    if provider == "claude":
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user_msg}])
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        return next((b.text for b in response.content if b.type == "text"), "")
    else:  # deepseek (OpenAI-compatible)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content or ""


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


def _discover_indicator(
    client,
    provider: str,
    tried: set,
    model: str,
) -> Optional[Dict]:
    """Ask LLM to suggest a new TradingView indicator not already tried."""
    tried_str = ", ".join(sorted(tried)) if tried else "none"
    user_msg = (
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
    )
    text = _llm_complete(client, provider, model, user_msg, max_tokens=512)
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
    client,
    provider: str,
    indicator: Dict[str, Any],
    cdc_source: str,
    model: str,
) -> Optional[str]:
    """Translate indicator description to Python using LLM."""
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
        "9. Output ONLY the Python code, no explanation, no markdown code blocks\n"
        f"10. MUST produce two boolean columns: {name}_bull and {name}_bear\n"
        f"    - {name}_bull = True when indicator is bullish (trend up / momentum positive)\n"
        f"    - {name}_bear = True when indicator is bearish (trend down / momentum negative)\n"
        "    - Both columns must be shift(1) like all other columns"
    )

    user_msg = (
        f"Translate this indicator to Python:\n"
        f"Name: {name}\n"
        f"Category: {category}\n"
        f"Description: {desc}\n"
        f"Default params: {json.dumps(params)}\n\n"
        f"Write the add_{name}_features(df, config) function."
    )

    code = _llm_complete(client, provider, model, user_msg, system=system_prompt)
    # Extract code from markdown code block if present (robust extraction)
    match = re.search(r'```(?:python)?\s*\n(.*?)```', code, flags=re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # Fallback: remove any stray backtick fences
        code = re.sub(r'```(?:python)?', '', code).strip()
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


def _debug_fix_code(
    client,
    provider: str,
    model: str,
    code: str,
    name: str,
    error_msg: str,
) -> Optional[str]:
    """Ask LLM to fix broken indicator code given the error message."""
    system_prompt = (
        "You are a Python debugging expert. Fix the provided code so it runs correctly.\n"
        "Rules:\n"
        f"1. Function name MUST be: add_{name}_features(df, config) -> pd.DataFrame\n"
        "2. All output columns shift(1) for lag safety\n"
        "3. No lookahead bias\n"
        "4. Include all imports at top (numpy, pandas, talib)\n"
        f"5. Return df copy with new columns prefixed: {name}_\n"
        f"6. MUST produce two boolean columns: {name}_bull and {name}_bear\n"
        "7. Output ONLY the fixed Python code, no explanation, no markdown code blocks\n"
    )
    user_msg = (
        f"This Python code for the '{name}' indicator has an error:\n\n"
        f"ERROR: {error_msg}\n\n"
        f"BROKEN CODE:\n{code}\n\n"
        f"Fix the code so it runs without errors. Output ONLY the corrected Python code."
    )
    fixed = _llm_complete(client, provider, model, user_msg, system=system_prompt)
    match = re.search(r'```(?:python)?\s*\n(.*?)```', fixed, flags=re.DOTALL)
    if match:
        fixed = match.group(1).strip()
    else:
        fixed = re.sub(r'```(?:python)?', '', fixed).strip()
    return fixed.strip() or None


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

def _get_tv_columns(code: str, name: str) -> List[str]:
    """Exec the indicator on a tiny DataFrame and return column names it adds."""
    import numpy as np
    import pandas as pd
    try:
        ns: Dict[str, Any] = {"np": np, "pd": pd}
        try:
            import talib; ns["talib"] = talib
        except ImportError:
            pass
        exec(compile(code, f"<tv_{name}>", "exec"), ns)  # noqa: S102
        fn = ns.get(f"add_{name}_features")
        if fn is None:
            return []
        np.random.seed(0)
        n = 300
        close = 1800.0 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            "Open": close - 1, "High": close + 2,
            "Low": close - 2, "Close": close,
            "Volume": np.ones(n) * 5000,
        })
        result = fn(df.copy(), {})
        return [c for c in result.columns if c not in df.columns]
    except Exception:
        return []


def _build_run_config(
    base_config: Dict,
    indicator: Dict,
    code: str = "",
    integration_mode: str = "hmm",
    filter_column_bull: str = "",
    filter_column_bear: str = "",
) -> Dict:
    """Return a deep-copy of base_config with the TV indicator enabled.

    integration_mode:
      "hmm"           — inject TV columns into HMM features only (original behavior)
      "signal_filter" — set integration_mode: signal_filter; skip HMM injection
      "both"          — inject into HMM features AND set signal_filter mode
    """
    cfg  = copy.deepcopy(base_config)
    name = indicator["name"]
    if "tv_indicators" not in cfg:
        cfg["tv_indicators"] = {}
    ind_entry: Dict[str, Any] = {
        "enabled": True,
        "integration_mode": integration_mode,
        **indicator.get("default_params", {}),
    }
    if filter_column_bull:
        ind_entry["filter_column_bull"] = filter_column_bull
    if filter_column_bear:
        ind_entry["filter_column_bear"] = filter_column_bear
    cfg["tv_indicators"][name] = ind_entry

    # Inject new columns into HMM feature list (only for hmm / both modes)
    if code and integration_mode in ("hmm", "both"):
        tv_cols = _get_tv_columns(code, name)
        if tv_cols:
            base_hmm_feats = [
                "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
                "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
                "hmm_feat_atr_ratio",
            ]
            existing = list(cfg.get("hmm", {}).get("features", base_hmm_feats))
            cfg.setdefault("hmm", {})["features"] = existing + tv_cols
            print(f"[tv] HMM features += {tv_cols[:4]}{'...' if len(tv_cols)>4 else ''}")

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


# Best-baseline config override (idea16: 43.6% return, 4H 2-state HMM)
_BEST_BASELINE_OVERRIDE: Dict[str, Any] = {
    "strategy": {"regime_timeframe": "4H"},
    "hmm":      {"n_states": 2, "sizing_max": 2.0},
    "risk":     {"base_allocation_frac": 0.90, "max_hold_bars": 48},
}


def _apply_best_baseline(config: Dict) -> Dict:
    """Deep-merge _BEST_BASELINE_OVERRIDE into config."""
    cfg = copy.deepcopy(config)
    for section, values in _BEST_BASELINE_OVERRIDE.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


# ---------------------------------------------------------------------------
# 3-mode per-indicator testing
# ---------------------------------------------------------------------------

def _test_three_modes(
    base_config: Dict,
    indicator: Dict,
    code: str,
    n_trials: int,
    walk_forward: bool,
    hmm_trained: bool,
    force_no_retrain: bool = False,
) -> Tuple[str, Dict, Dict[str, Dict]]:
    """Run pipeline in 3 modes and return (best_mode, best_metrics, all_results).

    Mode A: hmm           — TV columns injected into HMM features, retrain HMM
    Mode B: signal_filter — TV columns used as entry gate only, reuse existing HMM
    Mode C: both          — TV columns in HMM AND as entry gate, retrain HMM

    force_no_retrain: if True, all modes use retrain_model=False (reuse existing HMM).
    """
    name = indicator["name"]
    seed_integration = indicator.get("integration_mode", "hmm")
    filter_bull = indicator.get("filter_column_bull", "")
    filter_bear = indicator.get("filter_column_bear", "")

    all_results: Dict[str, Dict] = {}
    best_mode = "hmm"
    best_return = -999.0
    best_metrics: Dict = {}

    if force_no_retrain:
        modes = [
            ("hmm",           False),
            ("signal_filter", False),
            ("both",          False),
        ]
    else:
        modes = [
            ("hmm",           True),
            ("signal_filter", False if hmm_trained else True),
            ("both",          True),
        ]

    for mode, retrain in modes:
        print(f"[tv]   Mode {mode}: retrain={retrain}")
        try:
            run_cfg = _build_run_config(
                base_config, indicator, code,
                integration_mode=mode,
                filter_column_bull=filter_bull,
                filter_column_bear=filter_bear,
            )
            results     = _run_pipeline(run_cfg, n_trials, walk_forward, retrain)
            tm          = results.get("test_metrics", {})
            cur_return  = tm.get("annualized_return", 0)
            all_results[mode] = {
                "test_metrics": tm,
                "verdict":      results.get("verdict", "?"),
                "config":       {"tv_indicators": {name: run_cfg["tv_indicators"][name]}},
            }
            print(
                f"[tv]     -> return={cur_return*100:.1f}%"
                f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
                f" | verdict={results.get('verdict', '?')}"
            )
            if cur_return > best_return:
                best_return  = cur_return
                best_mode    = mode
                best_metrics = tm
        except Exception as e:
            all_results[mode] = {"error": str(e)}
            print(f"[tv]     -> FAILED: {e}")

    print(f"[tv]   Best mode: {best_mode} (return={best_return*100:.1f}%)")
    return best_mode, best_metrics, all_results


# ---------------------------------------------------------------------------
# Greedy stacking
# ---------------------------------------------------------------------------

def _greedy_stack(
    base_config: Dict,
    completed_entries: List[Dict],
    n_trials: int,
    walk_forward: bool,
) -> Dict[str, Any]:
    """Greedily stack the best-performing indicators.

    Algorithm:
    1. Rank completed entries by best individual return (descending).
    2. Start with an empty active set.
    3. For each indicator (best to worst): add to active set, run pipeline.
       Keep if return improves; drop otherwise.
    4. Return dict with final active set and combined metrics.
    """
    ranked = sorted(
        [e for e in completed_entries if e.get("test_metrics")],
        key=lambda x: x["test_metrics"].get("annualized_return", 0),
        reverse=True,
    )

    if not ranked:
        print("[tv] Greedy stack: no completed entries to stack.")
        return {}

    print(f"\n[tv] === Greedy Stacking ({len(ranked)} indicators) ===")

    active_set: Dict[str, Any] = {}  # name -> {integration_mode, filter_col_bull, filter_col_bear, ...}
    best_return = 0.0
    best_metrics: Dict = {}

    # Build a name -> entry lookup so we can find module paths for all active indicators
    entry_by_name = {e["name"]: e for e in ranked}

    for entry in ranked:
        name = entry["name"]
        # Get the best config override from 3-mode results if available
        mode_results = entry.get("mode_results", {})
        best_mode    = entry.get("best_mode", "hmm")
        ind_override = (
            mode_results.get(best_mode, {}).get("config", {}).get("tv_indicators", {}).get(name)
            or {"enabled": True, "integration_mode": best_mode}
        )

        # Build candidate active set
        candidate = dict(active_set)
        candidate[name] = ind_override

        # Build test config with all indicators in candidate set enabled
        test_cfg = copy.deepcopy(base_config)
        if "tv_indicators" not in test_cfg:
            test_cfg["tv_indicators"] = {}
        for ind_name, ind_params in candidate.items():
            test_cfg["tv_indicators"][ind_name] = ind_params

        # Rebuild HMM features: start from base and add all hmm/both indicator columns
        base_hmm_feats = list(base_config.get("hmm", {}).get("features", [
            "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
            "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
        ]))
        combined_feats = list(base_hmm_feats)
        for ind_name, ind_params in candidate.items():
            mode = ind_params.get("integration_mode", "hmm")
            if mode in ("hmm", "both"):
                ind_entry = entry_by_name.get(ind_name, {})
                code_path = ind_entry.get("python_module_path")
                if code_path:
                    full_path = Path(__file__).parents[3] / code_path
                    if full_path.exists():
                        ind_code = full_path.read_text()
                        tv_cols  = _get_tv_columns(ind_code, ind_name)
                        combined_feats.extend(tv_cols)
        test_cfg.setdefault("hmm", {})["features"] = combined_feats

        print(f"[tv]   Testing stack +{name} ({len(candidate)} total)...")
        try:
            results    = _run_pipeline(test_cfg, n_trials, walk_forward, retrain_model=True)
            tm         = results.get("test_metrics", {})
            cur_return = tm.get("annualized_return", 0)
            print(
                f"[tv]     -> return={cur_return*100:.1f}%"
                f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
            )
            if cur_return > best_return:
                best_return  = cur_return
                best_metrics = tm
                active_set   = candidate
                print(f"[tv]     -> KEPT {name} in stack (new best={cur_return*100:.1f}%)")
            else:
                print(f"[tv]     -> DROPPED {name} (no improvement)")
        except Exception as e:
            print(f"[tv]     -> FAILED: {e} (dropping {name})")

    print(f"\n[tv] Greedy stack complete: {len(active_set)} indicators kept")
    print(f"[tv]   Final return={best_return*100:.1f}%")
    print(f"[tv]   Active indicators: {list(active_set.keys())}")

    return {
        "active_set":    active_set,
        "final_metrics": best_metrics,
        "n_indicators":  len(active_set),
    }


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
    parser.add_argument("--provider",            default="claude", choices=["claude", "deepseek"],
                        help="Provider for indicator discovery (source=web)")
    parser.add_argument("--model",               default=None, help="Model for discovery provider")
    parser.add_argument("--translate-provider",  default=None, choices=["claude", "deepseek"],
                        help="Provider for Pine->Python translation (default: same as --provider)")
    parser.add_argument("--translate-model",     default=None, help="Model for translation provider")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--indicators",    default=None, help="Comma-separated specific indicators")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--goal-return",   type=float, default=None)
    parser.add_argument("--goal-sharpe",   type=float, default=None)
    parser.add_argument("--walk-forward",      action="store_true")
    parser.add_argument("--retrain-model",     action="store_true")
    parser.add_argument("--no-retrain",        action="store_true",
                        help="Use existing HMM model for all modes (skip retraining)")
    parser.add_argument("--base-config",       default="configs/base.yaml")
    parser.add_argument("--use-best-baseline", action="store_true",
                        help="Use 43.6%% return baseline config (4H 2-state HMM, idea16)")
    parser.add_argument("--skip-greedy-stack", action="store_true",
                        help="Skip greedy stacking step after individual tests")
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

    # Apply best-baseline override if requested
    if args.use_best_baseline:
        base_config = _apply_best_baseline(base_config)
        print("[tv] Using best-baseline override (4H 2-state HMM, idea16)")

    tv_cfg = base_config.get("tv_research", {})

    provider  = args.provider
    # CLI arg > config (only if config model matches this provider) > provider default
    if args.model:
        model = args.model
    elif tv_cfg.get("model") and provider == "claude":
        model = tv_cfg["model"]
    else:
        model = _DEFAULT_MODEL[provider]

    # Translation provider (can differ from discovery provider)
    t_provider = args.translate_provider or provider
    if args.translate_model:
        t_model = args.translate_model
    elif t_provider == "claude":
        t_model = tv_cfg.get("model", _DEFAULT_MODEL["claude"])
    else:
        t_model = _DEFAULT_MODEL[t_provider]
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

    # ---- Baseline metrics (best known result across all sources) ----
    baseline_metrics: Dict = {}
    baseline_source: str = "none"

    def _try_load_test_metrics(path: Path, key: str = "test_metrics") -> Optional[Dict]:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                # pick entry with highest test return
                candidates = [e.get(key, {}) for e in data if e.get(key)]
                return max(candidates, key=lambda m: m.get("annualized_return", -999)) if candidates else None
            return data.get(key)
        except Exception:
            return None

    # 1. final_verdict.json (last pipeline run)
    verdict_path = PROJECT_ROOT / "artefacts" / "reports" / "final_verdict.json"
    if verdict_path.exists():
        m = _try_load_test_metrics(verdict_path)
        if m and m.get("annualized_return", -999) > baseline_metrics.get("annualized_return", -999):
            baseline_metrics = m
            baseline_source = "final_verdict.json"

    # 2. full_scheme_search result files
    for fpath in sorted((PROJECT_ROOT / "artefacts").glob("full_scheme_search_results*.json")):
        m = _try_load_test_metrics(fpath)
        if m and m.get("annualized_return", -999) > baseline_metrics.get("annualized_return", -999):
            baseline_metrics = m
            baseline_source = fpath.name

    # 3. tv_research log (best previous TV run)
    tv_log_path = PROJECT_ROOT / "artefacts" / "tv_research" / "tv_research_best.json"
    if tv_log_path.exists():
        m = _try_load_test_metrics(tv_log_path)
        if m and m.get("annualized_return", -999) > baseline_metrics.get("annualized_return", -999):
            baseline_metrics = m
            baseline_source = "tv_research_best.json"

    if baseline_metrics:
        print(
            f"[tv] Baseline: return={baseline_metrics.get('annualized_return', 0)*100:.1f}%"
            f" | sharpe={baseline_metrics.get('sharpe_ratio', 0):.3f}"
            f" | source={baseline_source}"
        )

    # ---- LLM clients ----
    client    = _build_client(provider)
    t_client  = _build_client(t_provider) if t_provider != provider else client
    print(f"[tv] discovery:    provider={provider} | model={model}")
    print(f"[tv] translation:  provider={t_provider} | model={t_model}")

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
                indicator = _discover_indicator(client, provider, tried, model)
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
        print(f"[tv] Translating {name} to Python via {t_provider} ({t_model})...")
        try:
            code = _translate_to_python(t_client, t_provider, indicator, cdc_source, t_model)
            if not code:
                raise ValueError("Translation returned empty output")
        except Exception as e:
            entry.update(status="TRANSLATION_FAILED", error=str(e),
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[tv] TRANSLATION FAILED: {e}")
            continue

        # ---- Validate + test run (with self-debug retry loop) ----
        MAX_DEBUG_ATTEMPTS = 10
        code_ok = False
        last_error = ""
        for attempt in range(MAX_DEBUG_ATTEMPTS):
            attempt_label = f"attempt {attempt+1}/{MAX_DEBUG_ATTEMPTS}"

            valid, msg = _validate_code(code, name)
            if not valid:
                last_error = f"Validation: {msg}"
                print(f"[tv] CODE ERROR ({attempt_label}): {msg}")
                if attempt < MAX_DEBUG_ATTEMPTS - 1:
                    print(f"[tv] Auto-debugging {name}...")
                    fixed = _debug_fix_code(t_client, t_provider, t_model, code, name, last_error)
                    if fixed:
                        code = fixed
                    else:
                        print(f"[tv] Debug returned empty code, stopping retries.")
                        break
                continue

            ok, run_msg = _test_run(code, name)
            if not ok:
                last_error = f"TestRun: {run_msg}"
                print(f"[tv] TEST RUN FAILED ({attempt_label}): {run_msg}")
                if attempt < MAX_DEBUG_ATTEMPTS - 1:
                    print(f"[tv] Auto-debugging {name}...")
                    fixed = _debug_fix_code(t_client, t_provider, t_model, code, name, last_error)
                    if fixed:
                        code = fixed
                    else:
                        print(f"[tv] Debug returned empty code, stopping retries.")
                        break
                continue

            # Both checks passed
            print(f"[tv] Validation: {msg} ({attempt_label})")
            print(f"[tv] Test run: {run_msg}")
            code_ok = True
            break

        if not code_ok:
            entry.update(status="CODE_ERROR", error=last_error,
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[tv] CODE ERROR after {MAX_DEBUG_ATTEMPTS} attempts: {last_error}")
            continue

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

        # ---- Run 3-mode tests ----
        print(f"[tv] Testing 3 modes (optimize={n_trials > 0}, trials={n_trials})...")
        try:
            hmm_already_trained = (iteration > 0)
            best_mode, test_metrics, mode_results = _test_three_modes(
                base_config, indicator, code,
                n_trials, args.walk_forward, hmm_already_trained,
                force_no_retrain=args.no_retrain,
            )
            verdict = mode_results.get(best_mode, {}).get("verdict", "?")

            entry["status"]       = "COMPLETED"
            entry["test_metrics"] = test_metrics
            entry["verdict"]      = verdict
            entry["best_mode"]    = best_mode
            entry["mode_results"] = mode_results
            entry["config_override"] = mode_results.get(best_mode, {}).get("config")

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
                print(f"[tv] NEW BEST: {name} (mode={best_mode}) | return={cur_return*100:.1f}%")

            print(
                f"[tv] Done: best_mode={best_mode}"
                f" | return={cur_return*100:.1f}%"
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

    # ---- Greedy stacking ----
    if not args.dry_run and not args.skip_greedy_stack:
        completed = [e for e in log if e.get("status") == "COMPLETED" and e.get("test_metrics")]
        if len(completed) >= 2:
            stack_result = _greedy_stack(
                base_config, completed, n_trials, args.walk_forward,
            )
            stack_path = tv_research_dir / "tv_research_stack.json"
            _save_best(stack_path, stack_result)
            print(f"[tv] Stack result saved: {stack_path}")
        else:
            print("[tv] Greedy stack skipped: need at least 2 completed indicators.")

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
