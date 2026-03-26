"""
app/scalp_research_loop.py - Scalping Indicator Research Loop CLI.

Discovers scalping-suitable indicators (fast-reacting, momentum, mean-reversion),
translates them to Python, and backtests via the existing pipeline in multi-TF mode
(1H HMM regime + 5M signals) with tight 1:1.5 R:R scalping parameters.

Forked from tv_research_loop.py. Discovery is always LLM-driven (no seed list).

Usage:
    scalp_research
    scalp_research --max-ideas 20 --trials 30
    scalp_research --dry-run --max-ideas 2
    scalp_research --goal-return 0.40 --goal-sharpe 2.0
    scalp_research --min-trades-per-day 8
    scalp_research --walk-forward --no-retrain
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import ast
import copy
import itertools
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

SCALP_CONFIG_PATH    = PROJECT_ROOT / "configs" / "scalp.yaml"
SCALP_SEED_LIST_PATH = PROJECT_ROOT / "configs" / "scalp_seed_list.yaml"

SCALP_TECHNIQUES = {
    "momentum_breakout":    "Momentum breakout scalp - break of key level (resistance/support, intraday high/low) with volume confirmation",
    "vwap_pullback":        "VWAP pullback / trend-continuation scalp - price trends, pulls back to VWAP, enter on resumption",
    "range_mean_reversion": "Range / mean-reversion scalp - fade moves into support/resistance, bet price snaps back to range midpoint",
    "order_flow":           "Order-flow / tape-reading scalp - microstructure proxies from OHLCV: delta, bar imbalance, aggressive pressure",
    "opening_range":        "Opening-range / news-reaction scalp - first volatility burst after market open or catalyst; scalp first clean push",
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
    _max_retries: int = 8,
) -> str:
    """Unified LLM call: returns response text for Claude or DeepSeek.

    Retries on rate-limit errors with exponential backoff (up to _max_retries).
    """
    if provider == "claude":
        import anthropic as _anthropic
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user_msg}])
        if system:
            kwargs["system"] = system
        for attempt in range(_max_retries):
            try:
                with client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                return next((b.text for b in response.content if b.type == "text"), "")
            except _anthropic.RateLimitError as e:
                wait = min(60 * (2 ** attempt), 600)  # 60s, 120s, 240s ... max 600s
                print(f"[scalp] Claude rate limit hit (attempt {attempt+1}/{_max_retries}), waiting {wait}s...")
                time.sleep(wait)
            except _anthropic.APIStatusError as e:
                # Overload (529) or other server-side errors — back off and retry
                wait = min(30 * (2 ** attempt), 300)
                print(f"[scalp] Claude API status error {e.status_code} (attempt {attempt+1}/{_max_retries}), waiting {wait}s: {e}")
                time.sleep(wait)
            except _anthropic.APIConnectionError as e:
                wait = min(15 * (2 ** attempt), 120)
                print(f"[scalp] Claude connection error (attempt {attempt+1}/{_max_retries}), waiting {wait}s: {e}")
                time.sleep(wait)
        # final attempt (raises if still failing)
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
        msg = response.choices[0].message
        content = msg.content or ""
        # deepseek-reasoner spends tokens on reasoning; final answer may be in
        # reasoning_content when max_tokens is tight. Search both.
        if not content.strip():
            reasoning = getattr(msg, "reasoning_content", "") or ""
            # Extract the last JSON block from reasoning as the answer
            json_blocks = list(re.finditer(r'\{', reasoning))
            for match in reversed(json_blocks):
                start = match.start()
                depth = 0
                for i, ch in enumerate(reasoning[start:]):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = reasoning[start:start + i + 1]
                            try:
                                json.loads(candidate)
                                content = candidate
                            except json.JSONDecodeError:
                                pass
                            break
                if content.strip():
                    break
            if not content.strip():
                content = reasoning
        return content


# ---------------------------------------------------------------------------
# Log I/O helpers
# ---------------------------------------------------------------------------

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
# Seed list helpers
# ---------------------------------------------------------------------------

def _load_seed_list(seed_path: Path) -> List[Dict[str, Any]]:
    if not seed_path.exists():
        return []
    import yaml
    with open(seed_path) as f:
        data = yaml.safe_load(f)
    return data.get("indicators", [])


def _pick_from_seed(
    seed_list: List[Dict],
    tried: set,
    technique_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Pick the next untried indicator from the seed list.

    If technique_filter is set, only consider indicators matching that technique.
    Returns None when the seed list is exhausted.
    """
    for ind in seed_list:
        if ind["name"] in tried:
            continue
        if technique_filter and ind.get("technique") != technique_filter:
            continue
        return ind
    return None


# ---------------------------------------------------------------------------
# Scalping-specific indicator discovery
# ---------------------------------------------------------------------------

def _discover_scalp_indicator(
    client,
    provider: str,
    tried: set,
    model: str,
    technique_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Ask LLM to suggest a scalping-suitable indicator not already tried.

    Targets fast-reacting indicators suited for 5M timeframe scalping.
    If technique_filter is set, the LLM is constrained to that technique only.
    """
    tried_str = ", ".join(sorted(tried)) if tried else "none"

    techniques_block = "\n".join(
        f"  - {k}: {v}" for k, v in SCALP_TECHNIQUES.items()
    )

    if technique_filter:
        technique_instruction = (
            f"\nIMPORTANT: You MUST suggest an indicator for the '{technique_filter}' technique ONLY.\n"
            f"Technique description: {SCALP_TECHNIQUES.get(technique_filter, '')}\n"
        )
        technique_json = f'  "technique": "{technique_filter}",\n'
    else:
        technique_instruction = (
            "\nPick the most appropriate technique from the 5 options for your suggested indicator.\n"
        )
        technique_json = '  "technique": "momentum_breakout|vwap_pullback|range_mean_reversion|order_flow|opening_range",\n'

    user_msg = (
        "Suggest one TradingView indicator well-suited for SCALPING on 5-minute charts.\n\n"
        "The 5 scalping techniques to choose from:\n"
        f"{techniques_block}\n"
        f"{technique_instruction}\n"
        "Rules:\n"
        "1. Fast-reacting: responds within 3-14 bars (not slow trend indicators)\n"
        "2. Can be computed from OHLCV data with clear mathematical rules (not AI-based)\n"
        "3. AVOID: slow moving averages (period > 20), long-term trend followers, "
        "weekly/daily-level indicators\n"
        f"4. Has NOT been tried yet: [{tried_str}]\n\n"
        "Respond with ONLY a JSON object:\n"
        '{\n'
        f'{technique_json}'
        '  "name": "snake_case_name",\n'
        '  "category": "momentum|mean_reversion|micro_structure|volume|oscillator",\n'
        '  "description": "one sentence description",\n'
        '  "default_params": {"key": value},\n'
        '  "integration_mode": "feature"\n'
        '}'
    )
    text = _llm_complete(client, provider, model, user_msg, max_tokens=5000)
    if not text.strip():
        print("[scalp] WARNING: LLM returned empty text for discovery")
        return None
    print(f"[scalp] DEBUG discovery response (first 300 chars): {text[:300]!r}")
    # Try to extract JSON — handles nested braces in default_params
    for match in re.finditer(r'\{', text):
        start = match.start()
        depth = 0
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:start + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if "name" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Code translation + validation
# ---------------------------------------------------------------------------

def _translate_scalp_to_python(
    client,
    provider: str,
    indicator: Dict[str, Any],
    cdc_source: str,
    model: str,
) -> Optional[str]:
    """Translate scalping indicator description to Python using LLM.

    Enforces SHORT periods suitable for 5M scalping.
    """
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
        f"    - {name}_bull = True when indicator is bullish (momentum positive / oversold bounce)\n"
        f"    - {name}_bear = True when indicator is bearish (momentum negative / overbought reversal)\n"
        "    - Both columns must be shift(1) like all other columns\n"
        "11. SCALPING CONSTRAINT: Use SHORT periods suitable for 5M charts:\n"
        "    - Moving averages: period 3-8 bars\n"
        "    - Oscillators: period 3-10 bars\n"
        "    - Lookback windows: 5-12 bars\n"
        "    - Signal periods: 3-5 bars\n"
        "    Avoid periods > 20 for any calculation."
    )

    user_msg = (
        f"Translate this SCALPING indicator to Python:\n"
        f"Name: {name}\n"
        f"Category: {category}\n"
        f"Description: {desc}\n"
        f"Default params: {json.dumps(params)}\n\n"
        f"Write the add_{name}_features(df, config) function. "
        f"Remember: SHORT periods (3-14 bars max) for 5M scalping."
    )

    code = _llm_complete(client, provider, model, user_msg, system=system_prompt)
    match = re.search(r'```(?:python)?\s*\n(.*?)```', code, flags=re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
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
        namespace: Dict[str, Any] = {
            "np": np,
            "pd": pd,
            "__name__": f"__scalp_test_{name}__",
        }
        try:
            import talib
            namespace["talib"] = talib
        except ImportError:
            pass

        exec(compile(code, f"<scalp_{name}>", "exec"), namespace)  # noqa: S102

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
    header = (
        f'"""\n'
        f'features/tv_indicators/{name}.py - Auto-generated scalping indicator module.\n'
        f'Generated by scalp_research_loop on {datetime.now().isoformat()[:19]}\n'
        f'"""\n'
    )
    if code.startswith('"""') or code.startswith("'''"):
        full_code = code
    else:
        full_code = header + code

    with open(module_path, "w", encoding="utf-8") as f:
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
        exec(compile(code, f"<scalp_{name}>", "exec"), ns)  # noqa: S102
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

    Also forces multi_tf mode as a safety net in case scalp.yaml was
    not correctly merged.

    integration_mode:
      "hmm"              -- inject TV columns into HMM features only
      "signal_source"    -- TV columns OR'd with Donchian as entry trigger; no HMM injection
      "hmm_signal_source"-- inject into HMM features AND use as entry trigger
    """
    cfg  = copy.deepcopy(base_config)
    name = indicator["name"]

    # Safety net: force multi-TF mode
    cfg.setdefault("strategy", {})["mode"] = "multi_tf"

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

    # Inject new columns into HMM feature list (only for hmm / hmm_signal_source modes)
    if code and integration_mode in ("hmm", "hmm_signal_source"):
        tv_cols = _get_tv_columns(code, name)
        if tv_cols:
            base_hmm_feats = [
                "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
                "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
                "hmm_feat_atr_ratio",
            ]
            existing = list(cfg.get("hmm", {}).get("features", base_hmm_feats))
            cfg.setdefault("hmm", {})["features"] = existing + tv_cols
            print(f"[scalp] HMM features += {tv_cols[:4]}{'...' if len(tv_cols)>4 else ''}")

    return cfg


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    config: Dict,
    n_trials: int,
    walk_forward: bool,
    retrain_model: bool,
    model_path_override: str = None,
) -> Dict:
    from trade2.app.run_pipeline import run_pipeline
    return run_pipeline(
        config               = config,
        walk_forward         = walk_forward,
        retrain_model        = retrain_model,
        export_approved      = False,
        optimize             = n_trials > 0,
        n_trials             = n_trials,
        model_path_override  = model_path_override,
    )


# ---------------------------------------------------------------------------
# Goal check + trade frequency
# ---------------------------------------------------------------------------

def _goal_met(metrics: Dict, scalp_cfg: Dict) -> bool:
    goal_return = scalp_cfg.get("goal_return", 0.30)
    goal_sharpe = scalp_cfg.get("goal_sharpe", 1.5)
    goal_max_dd = scalp_cfg.get("goal_max_dd", -0.20)
    return (
        metrics.get("annualized_return", 0) >= goal_return
        and metrics.get("sharpe_ratio", 0)   >= goal_sharpe
        and metrics.get("max_drawdown", -999) >= goal_max_dd
    )


def _calc_trades_per_day(metrics: Dict) -> float:
    """Estimate trades per day from total_trades assuming ~252 trading days/year."""
    total_trades = metrics.get("total_trades", metrics.get("n_trades", 0))
    # Test period: 2025-01-01 to 2026-03-15 ~ 1.2 years (from base.yaml splits)
    n_years = 1.2
    if total_trades == 0:
        return 0.0
    return total_trades / (n_years * 252)


def _warn_low_frequency(trades_per_day: float, min_tpd: float) -> None:
    if trades_per_day < min_tpd:
        print(
            f"[scalp] WARNING: low trade frequency ({trades_per_day:.1f} trades/day, "
            f"target >= {min_tpd}). Consider relaxing HMM thresholds or session filter."
        )


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
    base_model_path: str = None,
) -> Tuple[str, Dict, Dict[str, Dict]]:
    """Run pipeline in 3 modes and return (best_mode, best_metrics, all_results).

    Mode A: signal_source     -- TV columns OR'd with Donchian as entry trigger; reuse HMM
    Mode B: hmm               -- TV columns injected into HMM features, retrain HMM
    Mode C: hmm_signal_source -- TV in HMM AND as entry trigger, retrain HMM
    """
    name = indicator["name"]
    filter_bull = indicator.get("filter_column_bull", "")
    filter_bear = indicator.get("filter_column_bear", "")

    all_results: Dict[str, Dict] = {}
    best_mode = "signal_source"
    best_return = -999.0
    best_metrics: Dict = {}

    if force_no_retrain:
        modes = [
            ("signal_source",     False, base_model_path),
            ("hmm",               False, None),
            ("hmm_signal_source", False, None),
        ]
    else:
        modes = [
            # signal_source reuses existing model — no retrain needed
            ("signal_source",     False, base_model_path),
            ("hmm",               True,  None),
            ("hmm_signal_source", True,  None),
        ]

    for mode, retrain, model_override in modes:
        print(f"[scalp]   Mode {mode}: retrain={retrain}"
              + (f" model={Path(model_override).name}" if model_override else ""))
        try:
            run_cfg = _build_run_config(
                base_config, indicator, code,
                integration_mode=mode,
                filter_column_bull=filter_bull,
                filter_column_bear=filter_bear,
            )
            results     = _run_pipeline(run_cfg, n_trials, walk_forward, retrain,
                                        model_path_override=model_override)
            tm          = results.get("test_metrics", {})
            cur_return  = tm.get("annualized_return", 0)
            tpd         = _calc_trades_per_day(tm)
            all_results[mode] = {
                "test_metrics":    tm,
                "verdict":         results.get("verdict", "?"),
                "trades_per_day":  tpd,
                "config":          {"tv_indicators": {name: run_cfg["tv_indicators"][name]}},
            }
            print(
                f"[scalp]     -> return={cur_return*100:.1f}%"
                f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
                f" | tpd={tpd:.1f}"
                f" | verdict={results.get('verdict', '?')}"
            )
            if cur_return > best_return:
                best_return  = cur_return
                best_mode    = mode
                best_metrics = tm
        except MemoryError as e:
            all_results[mode] = {"error": f"MemoryError: {e}", "status": "OOM"}
            print(f"[scalp]     -> OOM on mode {mode}: skipping")
        except Exception as e:
            all_results[mode] = {"error": str(e)}
            print(f"[scalp]     -> FAILED: {e}")

    print(f"[scalp]   Best mode: {best_mode} (return={best_return*100:.1f}%)")
    return best_mode, best_metrics, all_results


# ---------------------------------------------------------------------------
# Pair combo testing
# ---------------------------------------------------------------------------

def _test_pairs(
    base_config: Dict,
    completed_entries: List[Dict],
    n_trials: int,
    walk_forward: bool,
    top_n: int = 30,
) -> List[Dict]:
    """Test all pairs from the top N individual performers.

    Catches synergies between indicators that may not shine individually.
    Returns list of pair results sorted by return (best first).
    """
    ranked = sorted(
        [e for e in completed_entries if e.get("test_metrics") and e.get("python_module_path")],
        key=lambda x: x["test_metrics"].get("annualized_return", 0),
        reverse=True,
    )[:top_n]

    if len(ranked) < 2:
        print("[scalp] Pair testing: need at least 2 completed indicators.")
        return []

    pairs = list(itertools.combinations(ranked, 2))
    print(f"\n[scalp] === Pair Testing: {len(pairs)} pairs from top {len(ranked)} indicators ===")

    entry_by_name = {e["name"]: e for e in ranked}
    pair_results: List[Dict] = []

    for i, (a, b) in enumerate(pairs, 1):
        name_a, name_b = a["name"], b["name"]
        print(f"[scalp]   Pair {i}/{len(pairs)}: {name_a} + {name_b}")

        test_cfg = copy.deepcopy(base_config)
        test_cfg.setdefault("tv_indicators", {})

        # Use best mode from each indicator's individual test
        for entry in (a, b):
            name = entry["name"]
            best_mode = entry.get("best_mode", "signal_source")
            mode_results = entry.get("mode_results", {})
            ind_override = (
                mode_results.get(best_mode, {}).get("config", {}).get("tv_indicators", {}).get(name)
                or {"enabled": True, "integration_mode": best_mode or "signal_source"}
            )
            test_cfg["tv_indicators"][name] = ind_override

        # Build combined HMM features for any hmm-mode indicators
        base_hmm_feats = list(base_config.get("hmm", {}).get("features", [
            "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
            "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
            "hmm_feat_atr_ratio",
        ]))
        combined_feats = list(base_hmm_feats)
        for entry in (a, b):
            name = entry["name"]
            ind_params = test_cfg["tv_indicators"].get(name, {})
            mode = ind_params.get("integration_mode", "signal_source")
            if mode in ("hmm", "hmm_signal_source"):
                code_path = entry.get("python_module_path")
                if code_path:
                    full_path = Path(__file__).parents[3] / code_path
                    if full_path.exists():
                        tv_cols = _get_tv_columns(full_path.read_text(), name)
                        combined_feats.extend(tv_cols)
        test_cfg.setdefault("hmm", {})["features"] = combined_feats
        test_cfg.setdefault("strategy", {})["mode"] = "multi_tf"

        try:
            results = _run_pipeline(test_cfg, n_trials, walk_forward, retrain_model=False)
            tm = results.get("test_metrics", {})
            cur_return = tm.get("annualized_return", 0)
            tpd = _calc_trades_per_day(tm)
            print(
                f"[scalp]     -> return={cur_return*100:.1f}%"
                f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
                f" | tpd={tpd:.1f}"
            )
            pair_results.append({
                "pair":         (name_a, name_b),
                "test_metrics": tm,
                "trades_per_day": tpd,
                "config":       {name_a: test_cfg["tv_indicators"][name_a],
                                 name_b: test_cfg["tv_indicators"][name_b]},
            })
        except Exception as e:
            print(f"[scalp]     -> FAILED: {e}")

    pair_results.sort(key=lambda x: x.get("test_metrics", {}).get("annualized_return", 0), reverse=True)

    if pair_results:
        best = pair_results[0]
        print(
            f"\n[scalp] Best pair: {best['pair'][0]} + {best['pair'][1]}"
            f" | return={best['test_metrics'].get('annualized_return', 0)*100:.1f}%"
            f" | sharpe={best['test_metrics'].get('sharpe_ratio', 0):.3f}"
        )
        print(f"[scalp] Top 5 pairs:")
        for pr in pair_results[:5]:
            tm = pr["test_metrics"]
            print(
                f"[scalp]   {pr['pair'][0]:<30} + {pr['pair'][1]:<30}"
                f" ret={tm.get('annualized_return', 0)*100:.1f}%"
                f" sh={tm.get('sharpe_ratio', 0):.2f}"
            )

    return pair_results


# ---------------------------------------------------------------------------
# Greedy stacking
# ---------------------------------------------------------------------------

def _greedy_stack(
    base_config: Dict,
    completed_entries: List[Dict],
    n_trials: int,
    walk_forward: bool,
    seed_set: Dict[str, Any] = None,
    seed_return: float = 0.0,
) -> Dict[str, Any]:
    """Greedily stack the best-performing scalping indicators.

    Algorithm:
    1. Rank completed entries by best individual return (descending).
    2. Start from seed_set (best pair result) or empty set.
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
        print("[scalp] Greedy stack: no completed entries to stack.")
        return {}

    print(f"\n[scalp] === Greedy Stacking ({len(ranked)} indicators) ===")

    active_set: Dict[str, Any] = dict(seed_set) if seed_set else {}
    best_return = seed_return
    best_metrics: Dict = {}

    entry_by_name = {e["name"]: e for e in ranked}

    for entry in ranked:
        name = entry["name"]
        mode_results = entry.get("mode_results", {})
        best_mode    = entry.get("best_mode", "hmm")
        ind_override = (
            mode_results.get(best_mode, {}).get("config", {}).get("tv_indicators", {}).get(name)
            or {"enabled": True, "integration_mode": best_mode or "signal_source"}
        )

        candidate = dict(active_set)
        candidate[name] = ind_override

        test_cfg = copy.deepcopy(base_config)
        if "tv_indicators" not in test_cfg:
            test_cfg["tv_indicators"] = {}
        for ind_name, ind_params in candidate.items():
            test_cfg["tv_indicators"][ind_name] = ind_params

        base_hmm_feats = list(base_config.get("hmm", {}).get("features", [
            "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr", "hmm_feat_vol",
            "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
        ]))
        combined_feats = list(base_hmm_feats)
        for ind_name, ind_params in candidate.items():
            mode = ind_params.get("integration_mode", "hmm")
            if mode in ("hmm", "hmm_signal_source"):
                ind_entry = entry_by_name.get(ind_name, {})
                code_path = ind_entry.get("python_module_path")
                if code_path:
                    full_path = Path(__file__).parents[3] / code_path
                    if full_path.exists():
                        ind_code = full_path.read_text()
                        tv_cols  = _get_tv_columns(ind_code, ind_name)
                        combined_feats.extend(tv_cols)
        test_cfg.setdefault("hmm", {})["features"] = combined_feats

        # Safety net: force multi-TF mode
        test_cfg.setdefault("strategy", {})["mode"] = "multi_tf"

        print(f"[scalp]   Testing stack +{name} ({len(candidate)} total)...")
        try:
            results    = _run_pipeline(test_cfg, n_trials, walk_forward, retrain_model=True)
            tm         = results.get("test_metrics", {})
            cur_return = tm.get("annualized_return", 0)
            tpd        = _calc_trades_per_day(tm)
            print(
                f"[scalp]     -> return={cur_return*100:.1f}%"
                f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
                f" | tpd={tpd:.1f}"
            )
            if cur_return > best_return:
                best_return  = cur_return
                best_metrics = tm
                active_set   = candidate
                print(f"[scalp]     -> KEPT {name} in stack (new best={cur_return*100:.1f}%)")
            else:
                print(f"[scalp]     -> DROPPED {name} (no improvement)")
        except Exception as e:
            print(f"[scalp]     -> FAILED: {e} (dropping {name})")

    print(f"\n[scalp] Greedy stack complete: {len(active_set)} indicators kept")
    print(f"[scalp]   Final return={best_return*100:.1f}%")
    print(f"[scalp]   Active indicators: {list(active_set.keys())}")

    return {
        "active_set":    active_set,
        "final_metrics": best_metrics,
        "n_indicators":  len(active_set),
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _print_leaderboard(log: List[Dict]) -> None:
    completed = [
        e for e in log
        if e.get("status") in ("COMPLETED", "COMPLETED_LOW_FREQ") and e.get("test_metrics")
    ]
    if not completed:
        print("\n[scalp_research] No completed runs to rank.")
        return
    ranked = sorted(completed, key=lambda x: x["test_metrics"].get("annualized_return", 0), reverse=True)
    _TECH_ABBREV = {
        "momentum_breakout":    "mom_break",
        "vwap_pullback":        "vwap_pb",
        "range_mean_reversion": "mean_rev",
        "order_flow":           "ord_flow",
        "opening_range":        "open_rng",
        "unknown":              "unknown",
    }
    sep  = "=" * 102
    dash = "-" * 102
    print(f"\n{sep}")
    print("  SCALP INDICATOR RESEARCH - LEADERBOARD")
    print(sep)
    print(f"  {'#':<3} {'Indicator':<25} {'Technique':<12} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'TPD':>5} {'Verdict':<14}")
    print(dash)
    for i, entry in enumerate(ranked[:10], 1):
        tm = entry.get("test_metrics", {})
        tpd = entry.get("trades_per_day", _calc_trades_per_day(tm))
        tech = _TECH_ABBREV.get(entry.get("technique", "unknown"), entry.get("technique", "?")[:9])
        best_marker = " *" if entry.get("is_best") else ""
        low_freq_marker = " ~" if entry.get("status") == "COMPLETED_LOW_FREQ" else ""
        print(
            f"  {i:<3} {entry['name']:<25}"
            f" {tech:<12}"
            f" {tm.get('annualized_return', 0)*100:>7.1f}%"
            f" {tm.get('sharpe_ratio', 0):>8.3f}"
            f" {tm.get('max_drawdown', 0)*100:>7.1f}%"
            f" {tpd:>5.1f}"
            f"  {entry.get('verdict', '?'):<14}{best_marker}{low_freq_marker}"
        )
    print(f"{sep}")
    print("  (* = best so far, ~ = low trade frequency)")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scalping Indicator Research Loop")
    parser.add_argument("--max-ideas",          type=int,   default=30)
    parser.add_argument("--trials",             type=int,   default=50)
    parser.add_argument("--provider",           default="claude", choices=["claude", "deepseek"],
                        help="LLM provider for indicator discovery and translation")
    parser.add_argument("--model",              default=None, help="Override LLM model")
    parser.add_argument("--translate-provider", default=None, choices=["claude", "deepseek"],
                        help="Provider for Pine->Python translation (default: same as --provider)")
    parser.add_argument("--translate-model",    default=None, help="Model for translation provider")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Translate indicators only, skip pipeline execution")
    parser.add_argument("--skip-existing",      action="store_true")
    parser.add_argument("--goal-return",        type=float, default=None)
    parser.add_argument("--goal-sharpe",        type=float, default=None)
    parser.add_argument("--goal-max-dd",        type=float, default=None)
    parser.add_argument("--min-trades-per-day", type=float, default=None,
                        help="Minimum trades/day threshold for frequency warning")
    parser.add_argument("--walk-forward",       action="store_true")
    parser.add_argument("--no-retrain",         action="store_true",
                        help="Use existing HMM model for all modes (skip retraining)")
    parser.add_argument("--skip-greedy-stack",  action="store_true",
                        help="Skip greedy stacking step after individual tests")
    parser.add_argument("--skip-pairs",         action="store_true",
                        help="Skip pair combo testing, go straight to greedy stack")
    parser.add_argument("--top-pairs",          type=int, default=30,
                        help="Take top N individual performers for pair testing (default: 30)")
    parser.add_argument("--base-config",        default="configs/base.yaml")
    parser.add_argument("--scalp-config",       default="configs/scalp.yaml",
                        help="Path to scalp config overlay (e.g. configs/scalp_5m.yaml). Default: configs/scalp.yaml")
    parser.add_argument("--base-model-path",    default=None,
                        help="Path to HMM model used for signal_filter mode (default: artefacts/models/hmm_1h_3states.pkl)")
    parser.add_argument("--source",             default="both", choices=["seed", "llm", "both"],
                        help="Indicator source: seed=use seed list, llm=LLM discovery, both=seed first then LLM")
    parser.add_argument("--technique",          default=None, choices=list(SCALP_TECHNIQUES.keys()),
                        help="Filter to one scalping technique only")
    parser.add_argument("--seed-path",          default=None,
                        help="Path to a custom seed list yaml (overrides default scalp_seed_list.yaml)")
    parser.add_argument("--output-dir",         default=None,
                        help="Custom output directory for log/best/stack files (default: artefacts/scalp_research)")
    parser.add_argument("--base-model-id",      default="hmm_15m_3states",
                        help="Model filename stem for this instance (default: hmm_15m_3states). "
                             "Use unique values per parallel batch to avoid file collisions.")
    parser.add_argument("--module-prefix",      default="",
                        help="Prefix for generated module filenames to avoid collisions between "
                             "parallel runs (e.g. ds_ for DeepSeek, llm_ for LLM discovery).")
    args = parser.parse_args()

    # ---- Paths ----
    base_cfg_path      = PROJECT_ROOT / args.base_config
    scalp_research_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artefacts" / "scalp_research"
    tv_indicators_dir  = PROJECT_ROOT / "src" / "trade2" / "features" / "tv_indicators"
    scalp_research_dir.mkdir(parents=True, exist_ok=True)
    tv_indicators_dir.mkdir(parents=True, exist_ok=True)
    log_path   = scalp_research_dir / "scalp_research_log.json"
    best_path  = scalp_research_dir / "scalp_research_best.json"
    stack_path = scalp_research_dir / "scalp_research_stack.json"

    # ---- Load seed list ----
    seed_path = Path(args.seed_path) if args.seed_path else SCALP_SEED_LIST_PATH
    seed_list = _load_seed_list(seed_path)
    print(f"[scalp] Seed list: {len(seed_list)} indicators from {seed_path.name}")

    # ---- Load config: base.yaml merged with scalp config overlay ----
    from trade2.config.loader import load_config
    scalp_config_path = (PROJECT_ROOT / args.scalp_config) if not Path(args.scalp_config).is_absolute() else Path(args.scalp_config)
    if scalp_config_path.exists():
        base_config = load_config(str(base_cfg_path), str(scalp_config_path))
        print(f"[scalp] Config: {base_cfg_path.name} + {scalp_config_path.name}")
    else:
        base_config = load_config(str(base_cfg_path))
        print(f"[scalp] WARNING: scalp config not found at {scalp_config_path}, using base.yaml only")

    # Inject model_id so parallel batches write to isolated model files
    if args.base_model_id and args.base_model_id != "hmm_15m_3states":
        base_config.setdefault("hmm", {})["model_id"] = args.base_model_id

    scalp_cfg = base_config.get("scalp_research", {})

    # ---- Provider + model setup ----
    provider = args.provider
    if args.model:
        model = args.model
    elif scalp_cfg.get("model") and provider == "claude":
        model = scalp_cfg["model"]
    else:
        model = _DEFAULT_MODEL[provider]

    t_provider = args.translate_provider or scalp_cfg.get("translate_provider") or provider
    if args.translate_model:
        t_model = args.translate_model
    elif scalp_cfg.get("translate_model") and not args.translate_provider:
        t_model = scalp_cfg["translate_model"]
    elif t_provider == "claude":
        t_model = scalp_cfg.get("model", _DEFAULT_MODEL["claude"])
    else:
        t_model = _DEFAULT_MODEL[t_provider]

    max_ideas = args.max_ideas
    n_trials  = args.trials

    # ---- Goal overrides ----
    if args.goal_return is not None:
        scalp_cfg["goal_return"] = args.goal_return
    if args.goal_sharpe is not None:
        scalp_cfg["goal_sharpe"] = args.goal_sharpe
    if args.goal_max_dd is not None:
        scalp_cfg["goal_max_dd"] = args.goal_max_dd

    min_tpd = args.min_trades_per_day or scalp_cfg.get("min_trades_per_day", 5.0)

    # ---- Existing log ----
    log   = _load_log(log_path)
    # Always load previously tried names to avoid repeating across sessions
    tried = _get_tried_names(log)

    # ---- CDC template for translation prompt ----
    cdc_path   = PROJECT_ROOT / "src" / "trade2" / "features" / "cdc.py"
    cdc_source = cdc_path.read_text() if cdc_path.exists() else ""

    # ---- Baseline metrics (best from previous scalp runs) ----
    baseline_metrics: Dict = {}
    baseline_source: str = "none"

    if best_path.exists():
        try:
            with open(best_path) as f:
                best_data = json.load(f)
            tm = best_data.get("test_metrics", {})
            if tm.get("annualized_return", -999) > baseline_metrics.get("annualized_return", -999):
                baseline_metrics = tm
                baseline_source = "scalp_research_best.json"
        except Exception:
            pass

    if baseline_metrics:
        print(
            f"[scalp] Baseline: return={baseline_metrics.get('annualized_return', 0)*100:.1f}%"
            f" | sharpe={baseline_metrics.get('sharpe_ratio', 0):.3f}"
            f" | source={baseline_source}"
        )

    # ---- Base model for signal_filter mode ----
    # If --base-model-path given, use it directly.
    # Otherwise, train a fresh clean HMM (base features only, no TV indicator) once
    # and use it as the reference model for all signal_filter runs in this session.
    if args.base_model_path:
        base_model_path = str(PROJECT_ROOT / args.base_model_path) if not Path(args.base_model_path).is_absolute() else args.base_model_path
        if not Path(base_model_path).exists():
            print(f"[scalp] WARNING: --base-model-path not found at {base_model_path}, will retrain")
            base_model_path = None
        else:
            print(f"[scalp] Base model for signal_filter: {Path(base_model_path).name} (from --base-model-path)")
    elif args.dry_run:
        base_model_path = None  # dry-run skips pipeline entirely
    else:
        print("[scalp] Training fresh base HMM (clean 7-feature model for signal_source reference)...")
        try:
            import copy as _copy
            base_cfg_clean = _copy.deepcopy(base_config)
            # Strip any TV indicators so the base model trains on clean features only
            base_cfg_clean.pop("tv_indicators", None)
            _run_pipeline(base_cfg_clean, n_trials=0, walk_forward=False, retrain_model=True)
            _regime_tf   = base_cfg_clean.get("strategy", {}).get("regime_timeframe", "15M").lower()
            _n_states    = base_cfg_clean.get("hmm", {}).get("n_states", 3)
            _model_fname = f"hmm_{_regime_tf}_{_n_states}states_{args.base_model_id}.pkl"
            base_model_path = str(PROJECT_ROOT / "artefacts" / "models" / _model_fname)
            print(f"[scalp] Base model trained and saved to {Path(base_model_path).name}")
        except Exception as _e:
            print(f"[scalp] WARNING: base model training failed ({_e}), signal_source will reuse whatever is on disk")
            base_model_path = str(PROJECT_ROOT / "artefacts" / "models" / "hmm_15m_3states.pkl")
            if not Path(base_model_path).exists():
                base_model_path = None

    # ---- LLM clients ----
    client   = _build_client(provider)
    t_client = _build_client(t_provider) if t_provider != provider else client
    print(f"[scalp] discovery:   provider={provider} | model={model}")
    print(f"[scalp] translation: provider={t_provider} | model={t_model}")

    best_return = baseline_metrics.get("annualized_return", 0.0)
    best_entry: Optional[Dict] = None

    print(f"\n[scalp] Scalping Indicator Research Loop")
    print(f"[scalp] max_ideas={max_ideas} | model={model} | trials={n_trials} | dry_run={args.dry_run}")
    print(f"[scalp] source={args.source} | technique={args.technique or 'all'} | seed_count={len(seed_list)}")
    print(f"[scalp] strategy: 1H HMM regime + 5M signals | SL=1.0x ATR | TP=1.5x ATR")
    print(f"[scalp] goals: return>={scalp_cfg.get('goal_return', 0.30)*100:.0f}%"
          f" | sharpe>={scalp_cfg.get('goal_sharpe', 1.5):.1f}"
          f" | dd>={scalp_cfg.get('goal_max_dd', -0.20)*100:.0f}%"
          f" | tpd>={min_tpd:.1f}")
    print(f"[scalp] Techniques:")
    for tk, tdesc in SCALP_TECHNIQUES.items():
        marker = " <-- active" if args.technique == tk else ""
        print(f"[scalp]   {tk:<25} {tdesc[:60]}{marker}")
    print()

    for iteration in range(max_ideas):
        print(f"\n[scalp] === Iteration {iteration+1}/{max_ideas} ===")

        # ---- Pick indicator: seed list or LLM ----
        indicator: Optional[Dict] = None
        use_seed = args.source in ("seed", "both")
        use_llm  = args.source in ("llm", "both")

        if use_seed:
            indicator = _pick_from_seed(seed_list, tried, technique_filter=args.technique)
            if indicator is not None:
                indicator = dict(indicator)          # copy so we can mutate
                indicator["_source"] = "seed"
                print(f"[scalp] Source: seed | {indicator['name']} (technique={indicator.get('technique','?')})")

        if indicator is None and use_llm:
            print(f"[scalp] Source: LLM | Asking for scalping indicator (technique={args.technique or 'any'})...")
            for _disc_attempt in range(3):
                try:
                    indicator = _discover_scalp_indicator(
                        client, provider, tried, model,
                        technique_filter=args.technique,
                    )
                    if indicator is not None:
                        indicator["_source"] = "llm"
                        break
                    print(f"[scalp] Discovery returned no indicator (attempt {_disc_attempt+1}/3), retrying...")
                except Exception as e:
                    print(f"[scalp] Discovery error (attempt {_disc_attempt+1}/3): {e}")

        if indicator is None or "name" not in indicator:
            if args.source == "seed":
                print("[scalp] Seed list exhausted. Stopping loop.")
                break
            print("[scalp] Discovery failed after 3 attempts. Skipping iteration.")
            continue

        name = args.module_prefix + indicator["name"]
        tried.add(name)
        tech = indicator.get("technique", "unknown")
        print(f"[scalp] Indicator: {name} | technique={tech} | category={indicator.get('category', '?')}")
        print(f"[scalp]   {indicator.get('description', '')}")

        entry: Dict[str, Any] = {
            "id":                len(log) + 1,
            "name":              name,
            "category":          indicator.get("category", "unknown"),
            "technique":         indicator.get("technique", "unknown"),
            "source":            indicator.get("_source", "llm"),
            "integration_mode":  indicator.get("integration_mode", "feature"),
            "status":            "STARTED",
            "timestamp":         datetime.now().isoformat()[:19],
            "duration_seconds":  0,
            "python_module_path": None,
            "config_override":   None,
            "test_metrics":      None,
            "trades_per_day":    None,
            "baseline_metrics":  baseline_metrics,
            "delta_vs_baseline": None,
            "verdict":           None,
            "is_best":           False,
            "error":             None,
        }
        t0 = time.time()

        # ---- Translate to Python ----
        print(f"[scalp] Translating {name} to Python via {t_provider} ({t_model})...")
        try:
            code = _translate_scalp_to_python(t_client, t_provider, indicator, cdc_source, t_model)
            if not code:
                raise ValueError("Translation returned empty output")
        except Exception as e:
            entry.update(status="TRANSLATION_FAILED", error=str(e),
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[scalp] TRANSLATION FAILED: {e}")
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
                print(f"[scalp] CODE ERROR ({attempt_label}): {msg}")
                if attempt < MAX_DEBUG_ATTEMPTS - 1:
                    print(f"[scalp] Auto-debugging {name}...")
                    fixed = _debug_fix_code(t_client, t_provider, t_model, code, name, last_error)
                    if fixed:
                        code = fixed
                    else:
                        print(f"[scalp] Debug returned empty code, stopping retries.")
                        break
                continue

            ok, run_msg = _test_run(code, name)
            if not ok:
                last_error = f"TestRun: {run_msg}"
                print(f"[scalp] TEST RUN FAILED ({attempt_label}): {run_msg}")
                if attempt < MAX_DEBUG_ATTEMPTS - 1:
                    print(f"[scalp] Auto-debugging {name}...")
                    fixed = _debug_fix_code(t_client, t_provider, t_model, code, name, last_error)
                    if fixed:
                        code = fixed
                    else:
                        print(f"[scalp] Debug returned empty code, stopping retries.")
                        break
                continue

            print(f"[scalp] Validation: {msg} ({attempt_label})")
            print(f"[scalp] Test run: {run_msg}")
            code_ok = True
            break

        if not code_ok:
            entry.update(status="CODE_ERROR", error=last_error,
                         duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print(f"[scalp] CODE ERROR after {MAX_DEBUG_ATTEMPTS} attempts: {last_error}")
            continue

        # ---- Write module ----
        module_path = _write_module(code, name, tv_indicators_dir)
        entry["python_module_path"] = str(module_path.relative_to(PROJECT_ROOT))
        print(f"[scalp] Module written: {module_path.name}")

        if args.dry_run:
            entry.update(status="DRY_RUN", duration_seconds=int(time.time() - t0))
            log.append(entry)
            _save_log(log_path, log)
            print("[scalp] DRY RUN: pipeline skipped")
            continue

        # ---- Run 3-mode tests ----
        print(f"[scalp] Testing 3 modes (optimize={n_trials > 0}, trials={n_trials})...")
        try:
            hmm_already_trained = (iteration > 0)
            best_mode, test_metrics, mode_results = _test_three_modes(
                base_config, indicator, code,
                n_trials, args.walk_forward, hmm_already_trained,
                force_no_retrain=args.no_retrain,
                base_model_path=base_model_path,
            )
            verdict  = mode_results.get(best_mode, {}).get("verdict", "?")
            tpd      = _calc_trades_per_day(test_metrics)

            # Frequency check
            _warn_low_frequency(tpd, min_tpd)
            status = "COMPLETED_LOW_FREQ" if tpd < min_tpd else "COMPLETED"

            entry["status"]       = status
            entry["test_metrics"] = test_metrics
            entry["trades_per_day"] = tpd
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
                print(
                    f"[scalp] NEW BEST: {name} (mode={best_mode})"
                    f" | return={cur_return*100:.1f}%"
                    f" | tpd={tpd:.1f}"
                )

            print(
                f"[scalp] Done: best_mode={best_mode}"
                f" | return={cur_return*100:.1f}%"
                f" | sharpe={test_metrics.get('sharpe_ratio', 0):.3f}"
                f" | tpd={tpd:.1f}"
                f" | verdict={verdict}"
            )

            if _goal_met(test_metrics, scalp_cfg) and tpd >= min_tpd:
                print(f"\n[scalp] GOAL MET by {name}! Stopping loop.")
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
            print(f"[scalp] PIPELINE FAILED: {e}")

        entry["duration_seconds"] = int(time.time() - t0)
        log.append(entry)
        _save_log(log_path, log)

    # ---- Pair combo testing ----
    pair_seed_set: Dict[str, Any] = {}
    pair_seed_return: float = 0.0
    pair_results_path = scalp_research_dir / "scalp_pair_results.json"

    if not args.dry_run and not args.skip_pairs and not args.skip_greedy_stack:
        completed = [
            e for e in log
            if e.get("status") in ("COMPLETED", "COMPLETED_LOW_FREQ") and e.get("test_metrics")
        ]
        if len(completed) >= 2:
            pair_results = _test_pairs(
                base_config, completed, n_trials, args.walk_forward,
                top_n=args.top_pairs,
            )
            if pair_results:
                _save_best(pair_results_path, {"pairs": pair_results[:20]})
                print(f"[scalp] Pair results saved: {pair_results_path}")
                # Seed greedy stack from the best pair
                best_pair = pair_results[0]
                pair_seed_set = best_pair.get("config", {})
                pair_seed_return = best_pair["test_metrics"].get("annualized_return", 0)
        else:
            print("[scalp] Pair testing skipped: need at least 2 completed indicators.")

    # ---- Greedy stacking ----
    if not args.dry_run and not args.skip_greedy_stack:
        completed = [
            e for e in log
            if e.get("status") in ("COMPLETED", "COMPLETED_LOW_FREQ") and e.get("test_metrics")
        ]
        if len(completed) >= 2:
            stack_result = _greedy_stack(
                base_config, completed, n_trials, args.walk_forward,
                seed_set=pair_seed_set or None,
                seed_return=pair_seed_return,
            )
            _save_best(stack_path, stack_result)
            print(f"[scalp] Stack result saved: {stack_path}")
        else:
            print("[scalp] Greedy stack skipped: need at least 2 completed indicators.")

    # ---- Final summary ----
    _print_leaderboard(log)
    if best_entry:
        tm = best_entry.get("test_metrics", {})
        tpd = best_entry.get("trades_per_day", 0)
        print(
            f"\n[scalp] Best: {best_entry['name']}"
            f" | return={tm.get('annualized_return', 0)*100:.1f}%"
            f" | sharpe={tm.get('sharpe_ratio', 0):.3f}"
            f" | tpd={tpd:.1f}"
        )
    print(f"\n[scalp] Log:   {log_path}")
    print(f"[scalp] Best:  {best_path}")
    print(f"[scalp] Pairs: {pair_results_path}")


if __name__ == "__main__":
    main()
