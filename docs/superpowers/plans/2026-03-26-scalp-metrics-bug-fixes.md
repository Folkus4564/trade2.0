# Scalp Metrics Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs that corrupt all 1M scalp research metrics (n_years wrong by 60x, signal_source model path mismatch), then backfill 185 existing log entries with correct values.

**Architecture:** Three independent changes — (1) add `"1min"` to two dicts in `engine.py` so bars_per_year is computed correctly for 1M data, (2) fix the model path in `scalp_research_loop.py` to match `run_pipeline.py`'s naming convention, (3) a one-off script that mathematically corrects all existing log entries in-place with backups.

**Tech Stack:** Python 3.10+, pytest, pathlib, json, math.sqrt

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `code3.0/src/trade2/backtesting/engine.py` | Modify lines 296-301, 455 | Add `"1min"` to `_TF_SCALE` and `_TF_TO_FREQ_WF` |
| `code3.0/src/trade2/app/scalp_research_loop.py` | Modify line 1117 | Fix `base_model_path` naming |
| `code3.0/backfill_scalp_metrics.py` | Create | One-off backfill script |
| `code3.0/tests/test_engine_tf_scale.py` | Create | Tests for TF scale correctness |
| `code3.0/tests/test_backfill.py` | Create | Tests for backfill math |

---

## Task 1: Test that bars_per_year for "1min" is wrong (red)

**Files:**
- Create: `code3.0/tests/test_engine_tf_scale.py`

Context: `run_backtest()` in `engine.py` uses `_TF_SCALE` to compute `bars_per_year`. The function signature includes a `freq` parameter. We can unit-test `bars_per_year` by inspecting the metrics output: a backtest on synthetic data with `freq="1min"` should produce `n_years` that reflects 362,880 bars/year, not 6,048. Since `n_years = len(equity_curve) / bars_per_year`, with 362,880 bars of synthetic 1M data n_years should be ~1.0; with the bug it's ~60.

Run tests from the `code3.0/` directory:
```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/test_engine_tf_scale.py -v
```

- [ ] **Step 1: Write the failing test**

Create `code3.0/tests/test_engine_tf_scale.py`:

```python
"""Tests that bars_per_year is computed correctly for each timeframe frequency."""
import math
import numpy as np
import pandas as pd
import pytest
from trade2.backtesting.engine import run_backtest


def _make_signal_df(n: int, freq: str) -> pd.DataFrame:
    """Minimal OHLCV + signal columns for run_backtest."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    price = 2000.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "Open":  price,
        "High":  price + 0.5,
        "Low":   price - 0.5,
        "Close": price,
        "Volume": 100.0,
        "signal_long":  0,
        "signal_short": 0,
        "signal_size":  1.0,
        "atr":          1.0,
    }, index=ts)
    # Put one long signal so the engine doesn't short-circuit
    df.iloc[10, df.columns.get_loc("signal_long")] = 1
    return df


def _minimal_config():
    return {
        "strategy": {"name": "test", "mode": "single_tf"},
        "hmm": {"sizing_base": 1.0, "sizing_max": 1.0, "min_prob_hard": 0.0},
        "risk": {
            "atr_stop_mult": 1.0, "atr_tp_mult": 2.0, "base_allocation_frac": 0.01,
            "max_hold_bars": 0, "trailing_enabled": False, "break_even_atr_trigger": 0.0,
        },
        "costs": {
            "spread_pips": 0, "slippage_pips": 0, "commission_rt": 0.0,
            "spread_asian_mult": 1.0, "spread_vol_mult": 1.0, "spread_vol_atr_lookback": 1,
        },
        "backtest": {
            "init_cash": 10000, "risk_per_trade": 0.01, "contract_size_oz": 100,
            "use_linear_sizing": False,
        },
        "session": {"enabled": False, "allowed_hours_utc": list(range(24))},
    }


EXPECTED_BARS_PER_YEAR = {
    "1min":  252 * 24 * 60,   # 362880
    "5min":  252 * 24 * 12,   # 72576
    "15min": 252 * 24 * 4,    # 24192
    "1h":    252 * 24,         # 6048
}


@pytest.mark.parametrize("freq,expected_bpy", list(EXPECTED_BARS_PER_YEAR.items()))
def test_n_years_matches_bars_per_year(freq, expected_bpy):
    """n_years = n_bars / bars_per_year — verify bars_per_year is correct for each freq."""
    n_bars = expected_bpy  # Exactly 1 year worth of bars at correct scale
    df = _make_signal_df(n_bars, freq)
    cfg = _minimal_config()
    metrics, _ = run_backtest(df, cfg, freq=freq, period_label="test", init_cash=10000)
    # n_years should be ~1.0 (within 1%) if bars_per_year is right
    assert abs(metrics["n_years"] - 1.0) < 0.01, (
        f"freq={freq}: expected n_years~1.0, got {metrics['n_years']:.4f} "
        f"(bars_per_year was {n_bars / metrics['n_years']:.0f}, expected {expected_bpy})"
    )
```

- [ ] **Step 2: Run test to confirm it fails for "1min"**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/test_engine_tf_scale.py -v -k "1min"
```

Expected: FAIL — `n_years~60.0` instead of `~1.0` (because `"1min"` falls through to default scale=1).
The 5min/15min/1h tests should pass (they're already in `_TF_SCALE`).

---

## Task 2: Fix `_TF_SCALE` in engine.py (green)

**Files:**
- Modify: `code3.0/src/trade2/backtesting/engine.py:296-301`

- [ ] **Step 1: Add "1min" entries to `_TF_SCALE`**

In `engine.py` at line 296, replace:
```python
    _TF_SCALE = {
        "5min": 12, "5m": 12, "5M": 12,
        "15min": 4, "15m": 4, "15M": 4,
        "30min": 2, "30m": 2, "30M": 2,
        "1h": 1, "1H": 1, "4h": 0,  # 4H: scale=0 (same bars, different unit)
    }
```
with:
```python
    _TF_SCALE = {
        "1min": 60, "1m": 60, "1M": 60,  # 1-minute: 60x more bars than 1H
        "5min": 12, "5m": 12, "5M": 12,
        "15min": 4, "15m": 4, "15M": 4,
        "30min": 2, "30m": 2, "30M": 2,
        "1h": 1, "1H": 1, "4h": 0,  # 4H: scale=0 (same bars, different unit)
    }
```

- [ ] **Step 2: Fix `_TF_TO_FREQ_WF` in the same file**

At line 455, replace:
```python
    _TF_TO_FREQ_WF = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
```
with:
```python
    _TF_TO_FREQ_WF = {"1M": "1min", "5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
```

- [ ] **Step 3: Run the test again — should pass now**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/test_engine_tf_scale.py -v
```

Expected: All 4 parametrized cases PASS.

- [ ] **Step 4: Run full test suite to verify no regressions**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/ -v --tb=short
```

Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/backtesting/engine.py code3.0/tests/test_engine_tf_scale.py
git commit -m "fix: add 1min to _TF_SCALE and _TF_TO_FREQ_WF in engine.py

bars_per_year for 1M data was 6048 (treated as 1H) instead of
362880, making n_years 60x too large and all annualized metrics wrong.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Fix signal_source model path in scalp_research_loop.py

**Files:**
- Modify: `code3.0/src/trade2/app/scalp_research_loop.py:1117`

Context: After `_run_pipeline()` trains the base HMM, the code sets `base_model_path` using just `{args.base_model_id}.pkl` (e.g. `batch_05.pkl`). But `run_pipeline.py` saves the model as `hmm_{regime_tf}_{n_states}states_{model_id}.pkl` (e.g. `hmm_15m_3states_batch_05.pkl`). We derive the filename from the same config keys `run_pipeline.py` uses.

- [ ] **Step 1: Replace line 1117 in scalp_research_loop.py**

Find:
```python
            base_model_path = str(PROJECT_ROOT / "artefacts" / "models" / f"{args.base_model_id}.pkl")
```
Replace with:
```python
            _regime_tf   = base_cfg_clean.get("strategy", {}).get("regime_timeframe", "15M").lower()
            _n_states    = base_cfg_clean.get("hmm", {}).get("n_states", 3)
            _model_fname = f"hmm_{_regime_tf}_{_n_states}states_{args.base_model_id}.pkl"
            base_model_path = str(PROJECT_ROOT / "artefacts" / "models" / _model_fname)
```

- [ ] **Step 2: Verify the fix looks correct — spot check**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -c "
from pathlib import Path
base_cfg = {'strategy': {'regime_timeframe': '15M'}, 'hmm': {'n_states': 3}}
tf = base_cfg.get('strategy', {}).get('regime_timeframe', '15M').lower()
n  = base_cfg.get('hmm', {}).get('n_states', 3)
print(f'hmm_{tf}_{n}states_batch_05.pkl')
"
```

Expected output: `hmm_15m_3states_batch_05.pkl`

- [ ] **Step 3: Commit**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/src/trade2/app/scalp_research_loop.py
git commit -m "fix: derive base_model_path from config in scalp_research_loop

Was constructing path as batch_05.pkl but run_pipeline saves as
hmm_15m_3states_batch_05.pkl. Now derives tf and n_states from config
to match run_pipeline's naming convention. Fixes signal_source mode.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Write and test the backfill math

**Files:**
- Create: `code3.0/tests/test_backfill.py`
- Create: `code3.0/backfill_scalp_metrics.py`

Context: We need to correct metrics stored in JSON log files. The correction factor is `scale=60` (bars_per_year_new / bars_per_year_old). All corrections are pure math on stored values — no backtests re-run.

Key log entry structure (from a real batch_09 entry):
```
entry.test_metrics = {
    "annualized_return": -0.0014,   # wrong — too small
    "total_return": -0.0923,        # CORRECT — unchanged
    "annualized_volatility": 0.0056,
    "sharpe_ratio": -7.4387,
    "sortino_ratio": -2.4311,
    "max_drawdown": -0.1048,        # CORRECT — unchanged
    "calmar_ratio": -0.0131,
    "n_years": 70.22,               # wrong — should be ~1.17
    "benchmark_return": 0.0082,     # wrong — annualized with wrong n_years
    "alpha_vs_benchmark": -0.0095,
    "information_ratio": -0.3024,   # cannot fix — bench_vol not stored
    "random_baseline": {
        "random_median_sharpe": 0.1881,
        "random_median_return": 0.0037,
        "random_p95_sharpe": 0.3176,
        "n_simulations": 200
    },
    "beats_random_baseline": False,
    "cost_sensitivity_2x": {
        "sharpe_ratio": -7.616,
        "annualized_return": -0.0029,
        "max_drawdown": -0.1916
    }
}
```

The backfill logic lives in a helper function `patch_metrics(m, scale)` which we test in isolation before wiring into the script.

- [ ] **Step 1: Write tests for the patch_metrics function**

Create `code3.0/tests/test_backfill.py`:

```python
"""Tests for the backfill math in backfill_scalp_metrics.py."""
import math
import pytest
import sys
import importlib
from pathlib import Path

# Add code3.0 root to path so we can import the script directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_patch():
    """Import patch_metrics from the script (deferred so tests fail clearly if missing)."""
    import backfill_scalp_metrics as bfm
    return bfm.patch_metrics


SCALE = 60
SQRT_SCALE = math.sqrt(60)


class TestPatchMetrics:
    """Unit tests for patch_metrics(m, scale=60)."""

    def _sample(self):
        """Real values from rsi_extreme_snapback batch_09 entry."""
        return {
            "annualized_return": -0.0014,
            "total_return": -0.0923,
            "annualized_volatility": 0.0056,
            "sharpe_ratio": -7.4387,
            "sortino_ratio": -2.4311,
            "max_drawdown": -0.1048,
            "calmar_ratio": -0.0131,
            "n_years": 70.22,
            "benchmark_return": 0.0082,
            "alpha_vs_benchmark": -0.0095,
            "information_ratio": -0.3024,
            "random_baseline": {
                "random_median_sharpe": 0.1881,
                "random_median_return": 0.0037,
                "random_p95_sharpe": 0.3176,
                "n_simulations": 200,
            },
            "beats_random_baseline": False,
            "cost_sensitivity_2x": {
                "sharpe_ratio": -7.616,
                "annualized_return": -0.0029,
                "max_drawdown": -0.1916,
            },
        }

    def test_n_years_corrected(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["n_years"] - 70.22 / 60) < 0.001

    def test_annualized_return_corrected(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected = (1 + (-0.0923)) ** (1 / (70.22 / 60)) - 1
        assert abs(result["annualized_return"] - expected) < 1e-6

    def test_total_return_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["total_return"] == -0.0923

    def test_max_drawdown_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["max_drawdown"] == -0.1048

    def test_sharpe_scaled_by_sqrt_scale(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["sharpe_ratio"] - (-7.4387 * SQRT_SCALE)) < 1e-4

    def test_ann_vol_scaled_by_sqrt_scale(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["annualized_volatility"] - (0.0056 * SQRT_SCALE)) < 1e-6

    def test_calmar_recomputed(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected_calmar = result["annualized_return"] / abs(-0.1048)
        assert abs(result["calmar_ratio"] - expected_calmar) < 1e-6

    def test_alpha_vs_benchmark_recomputed(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["alpha_vs_benchmark"] - (result["annualized_return"] - result["benchmark_return"])) < 1e-6

    def test_information_ratio_nulled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["information_ratio"] is None
        assert result["information_ratio_backfill_skipped"] is True

    def test_random_median_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["random_baseline"]["random_median_sharpe"] - (0.1881 * SQRT_SCALE)) < 1e-4

    def test_random_p95_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["random_baseline"]["random_p95_sharpe"] - (0.3176 * SQRT_SCALE)) < 1e-4

    def test_beats_random_re_evaluated(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected = result["sharpe_ratio"] > result["random_baseline"]["random_median_sharpe"]
        assert result["beats_random_baseline"] == expected

    def test_cost_sensitivity_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["cost_sensitivity_2x"]["sharpe_ratio"] - (-7.616 * SQRT_SCALE)) < 1e-4

    def test_cost_sensitivity_max_dd_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["cost_sensitivity_2x"]["max_drawdown"] == -0.1916

    def test_sortino_guard_zero(self):
        """patch_metrics must not raise when sortino_ratio is 0."""
        patch = _import_patch()
        m = self._sample()
        m["sortino_ratio"] = 0.0
        result = patch(m, SCALE)  # should not raise
        assert result["sortino_ratio"] == 0.0

    def test_sortino_guard_near_zero_numerator(self):
        """patch_metrics must not raise when ann_ret_old ~= rfr (0.04)."""
        patch = _import_patch()
        m = self._sample()
        m["annualized_return"] = 0.04  # numerator for down_std recovery = 0
        m["sortino_ratio"] = 1.0
        result = patch(m, SCALE)  # should not raise
        assert result["sortino_ratio"] == 1.0  # left unchanged

    def test_spot_check_n_years(self):
        """Regression: rsi_extreme_snapback n_years_new = 70.22/60 = 1.1703."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["n_years"] - 1.1703) < 0.001

    def test_spot_check_annualized_return(self):
        """Regression: rsi_extreme_snapback ann_ret_new ≈ -7.9%."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["annualized_return"] - (-0.079)) < 0.001

    def test_spot_check_sharpe(self):
        """Regression: rsi_extreme_snapback sharpe_new ≈ -57.6."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["sharpe_ratio"] - (-57.6)) < 0.5
```

- [ ] **Step 2: Run tests to confirm they fail (module not yet created)**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/test_backfill.py -v --tb=short 2>&1 | head -20
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'backfill_scalp_metrics'`.

---

## Task 5: Write the backfill script

**Files:**
- Create: `code3.0/backfill_scalp_metrics.py`

- [ ] **Step 1: Create the script**

Create `code3.0/backfill_scalp_metrics.py`:

```python
"""
backfill_scalp_metrics.py — One-off script to correct 1M-timeframe metric errors.

Bug: engine.py _TF_SCALE was missing "1min", so bars_per_year = 6048 (1H scale)
instead of 362880 (1M scale). All annualized metrics are wrong by a factor
derivable from scale = 362880 / 6048 = 60.

Run once after applying the engine.py fix:
    cd code3.0
    python backfill_scalp_metrics.py

Backs up every modified file to *_backup_prebackfill.json before writing.
"""
import json
import math
import shutil
import glob
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCALP_RESEARCH_DIR = PROJECT_ROOT / "artefacts" / "scalp_research"
BEST_FILE = SCALP_RESEARCH_DIR / "scalp_research_best.json"
SCALE = 60
SQRT_SCALE = math.sqrt(SCALE)
RFR = 0.04  # risk-free rate used by metrics engine


def patch_metrics(m: dict, scale: int = SCALE) -> dict:
    """
    Return a corrected copy of a test_metrics dict.
    Applies closed-form corrections for scale factor between
    wrong bars_per_year (6048) and correct bars_per_year (362880).
    """
    if not m or m.get("n_years", 0) <= 0:
        return m

    sqrt_s = math.sqrt(scale)
    m = dict(m)  # shallow copy — we'll replace nested dicts below too

    n_years_old = m["n_years"]
    n_years_new = n_years_old / scale
    total_return = m.get("total_return", 0.0)

    # --- primary scalars ---
    m["n_years"] = n_years_new
    m["annualized_return"] = (1 + total_return) ** (1 / n_years_new) - 1
    m["annualized_volatility"] = m.get("annualized_volatility", 0.0) * sqrt_s
    m["sharpe_ratio"] = m.get("sharpe_ratio", 0.0) * sqrt_s

    # sortino: reconstruct down_std_old then rescale
    sortino_old = m.get("sortino_ratio", 0.0)
    ann_ret_old = (1 + total_return) ** (1 / n_years_old) - 1  # old value
    if sortino_old != 0.0 and abs(ann_ret_old - RFR) >= 1e-9:
        down_std_old = (ann_ret_old - RFR) / sortino_old
        down_std_new = down_std_old * sqrt_s
        m["sortino_ratio"] = (m["annualized_return"] - RFR) / down_std_new
    # else: leave sortino as-is (guard cases)

    # calmar
    max_dd = m.get("max_drawdown", 0.0)
    if max_dd != 0.0:
        m["calmar_ratio"] = m["annualized_return"] / abs(max_dd)

    # benchmark
    bench_ann_old = m.get("benchmark_return", 0.0)
    bench_total = (1 + bench_ann_old) ** n_years_old - 1
    bench_ann_new = (1 + bench_total) ** (1 / n_years_new) - 1
    m["benchmark_return"] = bench_ann_new

    # alpha
    m["alpha_vs_benchmark"] = m["annualized_return"] - bench_ann_new

    # information_ratio: bench_vol not stored, cannot recover
    m["information_ratio"] = None
    m["information_ratio_backfill_skipped"] = True

    # random baseline (nested dict)
    rb = m.get("random_baseline")
    if rb and isinstance(rb, dict):
        rb = dict(rb)
        rb["random_median_sharpe"] = rb.get("random_median_sharpe", 0.0) * sqrt_s
        rb["random_p95_sharpe"]    = rb.get("random_p95_sharpe", 0.0) * sqrt_s
        # random_median_return: treat as annualized, derive from pseudo total_return
        old_rand_ret = rb.get("random_median_return", 0.0)
        pseudo_total = (1 + old_rand_ret) ** n_years_old - 1
        rb["random_median_return"] = (1 + pseudo_total) ** (1 / n_years_new) - 1
        m["random_baseline"] = rb

    # beats_random
    rb_new = m.get("random_baseline") or {}
    m["beats_random_baseline"] = m["sharpe_ratio"] > rb_new.get("random_median_sharpe", 0.0)

    # cost_sensitivity_2x (nested)
    cs = m.get("cost_sensitivity_2x")
    if cs and isinstance(cs, dict):
        cs = dict(cs)
        cs["sharpe_ratio"]       = cs.get("sharpe_ratio", 0.0) * sqrt_s
        old_cs_ret = cs.get("annualized_return", 0.0)
        cs_total = (1 + old_cs_ret) ** n_years_old - 1
        cs["annualized_return"]  = (1 + cs_total) ** (1 / n_years_new) - 1
        # max_drawdown unchanged
        m["cost_sensitivity_2x"] = cs

    return m


def _should_patch(entry: dict) -> bool:
    """Return True if this log entry needs the scale=60 correction."""
    # Must have real metrics
    tm = entry.get("test_metrics") or {}
    if not tm or tm.get("n_years", 0) <= 0:
        return False
    # Skip failed runs
    if entry.get("status") in ("TRANSLATION_FAILED", "CODE_ERROR"):
        return False
    # Only patch 1M signal timeframe (or entries with no explicit TF, which default to 1M)
    sig_tf = (
        entry.get("config_override", {})
             .get("strategy", {})
             .get("signal_timeframe", "1M")
    )
    if sig_tf != "1M":
        print(f"  SKIP (signal_tf={sig_tf}): {entry.get('name','?')}")
        return False
    return True


def patch_entry(entry: dict) -> dict:
    """Patch all metric objects in a single log entry."""
    entry = dict(entry)

    # Primary test_metrics
    if entry.get("test_metrics"):
        entry["test_metrics"] = patch_metrics(entry["test_metrics"])

    # baseline_metrics
    if entry.get("baseline_metrics"):
        entry["baseline_metrics"] = patch_metrics(entry["baseline_metrics"])

    # mode_results[*].test_metrics
    mode_results = entry.get("mode_results") or {}
    if mode_results:
        patched_modes = {}
        for mode_name, mode_val in mode_results.items():
            if isinstance(mode_val, dict) and mode_val.get("test_metrics"):
                mode_val = dict(mode_val)
                mode_val["test_metrics"] = patch_metrics(mode_val["test_metrics"])
            patched_modes[mode_name] = mode_val
        entry["mode_results"] = patched_modes

    entry["_backfill_applied"] = True
    return entry


def process_log_file(log_path: Path) -> tuple[int, int]:
    """Patch a single scalp_research_log.json. Returns (n_patched, n_total)."""
    with open(log_path) as f:
        data = json.load(f)

    runs = data if isinstance(data, list) else data.get("runs", [])
    n_patched = 0
    patched_runs = []

    for entry in runs:
        if _should_patch(entry) and not entry.get("_backfill_applied"):
            old_ret = (entry.get("test_metrics") or {}).get("annualized_return", "N/A")
            old_sh  = (entry.get("test_metrics") or {}).get("sharpe_ratio", "N/A")
            old_ny  = (entry.get("test_metrics") or {}).get("n_years", "N/A")
            entry = patch_entry(entry)
            new_ret = entry["test_metrics"]["annualized_return"]
            new_sh  = entry["test_metrics"]["sharpe_ratio"]
            new_ny  = entry["test_metrics"]["n_years"]
            name = entry.get("name") or entry.get("indicator_name", "?")
            print(f"  PATCHED {name:<40} n_years {old_ny:.1f}->{new_ny:.2f}  "
                  f"ret {old_ret:.4f}->{new_ret:.4f}  sh {old_sh:.2f}->{new_sh:.2f}")
            n_patched += 1
        patched_runs.append(entry)

    if n_patched == 0:
        return 0, len(runs)

    # Backup original
    backup_path = log_path.with_suffix(".json_backup_prebackfill.json")
    if not backup_path.exists():
        shutil.copy2(log_path, backup_path)
        print(f"  Backed up to {backup_path.name}")

    # Write patched
    if isinstance(data, list):
        out = patched_runs
    else:
        data["runs"] = patched_runs
        out = data

    with open(log_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    return n_patched, len(runs)


def rebuild_best(all_entries: list[dict]) -> dict | None:
    """Return the globally best entry by annualized_return."""
    candidates = [
        e for e in all_entries
        if (e.get("test_metrics") or {}).get("annualized_return", -999) > -999
        and e.get("status") not in ("TRANSLATION_FAILED", "CODE_ERROR")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["test_metrics"]["annualized_return"])


def main():
    print(f"Backfilling scalp research logs in: {SCALP_RESEARCH_DIR}")
    print(f"Scale factor: {SCALE}  (bars_per_year 6048 -> 362880)")
    print()

    log_files = sorted(SCALP_RESEARCH_DIR.glob("*/scalp_research_log.json"))
    if not log_files:
        print("No log files found.")
        return

    total_patched = 0
    total_runs = 0
    all_entries = []

    for log_path in log_files:
        batch = log_path.parent.name
        print(f"[{batch}] {log_path.name}")
        n_p, n_t = process_log_file(log_path)
        total_patched += n_p
        total_runs    += n_t
        # Reload for best-tracking
        with open(log_path) as f:
            d = json.load(f)
        runs = d if isinstance(d, list) else d.get("runs", [])
        all_entries.extend(runs)
        print()

    print(f"Summary: {total_patched} / {total_runs} entries patched across {len(log_files)} files.")
    print()

    # Rebuild best.json
    if BEST_FILE.exists():
        best_backup = BEST_FILE.with_suffix(".json_backup_prebackfill.json")
        if not best_backup.exists():
            shutil.copy2(BEST_FILE, best_backup)
    best = rebuild_best(all_entries)
    if best:
        with open(BEST_FILE, "w") as f:
            json.dump(best, f, indent=2, default=str)
        print(f"scalp_research_best.json updated: {best.get('name','?')} "
              f"ret={best['test_metrics']['annualized_return']:.1%} "
              f"sh={best['test_metrics']['sharpe_ratio']:.2f}")
    else:
        print("No valid entries for best.json.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tests — should now all pass**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/test_backfill.py -v
```

Expected: All 18 tests PASS.

- [ ] **Step 3: Commit script and tests**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/backfill_scalp_metrics.py code3.0/tests/test_backfill.py
git commit -m "feat: add backfill_scalp_metrics.py to correct 1M n_years bug in logs

Applies scale=60 correction to annualized metrics in all 185 existing
scalp_research_log.json entries. Backs up originals before overwriting.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Run the backfill and verify results

**Files:** (no code changes — execution only)

- [ ] **Step 1: Run the backfill script**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python backfill_scalp_metrics.py
```

Expected output: Each COMPLETED/COMPLETED_LOW_FREQ entry prints a line showing:
```
PATCHED rsi_extreme_snapback                  n_years 70.2->1.17  ret -0.0014->-0.0793  sh -7.44->-57.60
```

- [ ] **Step 2: Spot-check rsi_extreme_snapback (batch_09)**

```bash
python -c "
import json
with open('artefacts/scalp_research/batch_09/scalp_research_log.json') as f:
    d = json.load(f)
runs = d if isinstance(d, list) else d.get('runs', [])
r = runs[0]
m = r['test_metrics']
print('n_years:', m['n_years'])
print('annualized_return:', m['annualized_return'])
print('sharpe_ratio:', m['sharpe_ratio'])
print('beats_random:', m['beats_random_baseline'])
print('IR skipped:', m.get('information_ratio_backfill_skipped'))
print('backups exist:', __import__('pathlib').Path('artefacts/scalp_research/batch_09/scalp_research_log.json_backup_prebackfill.json').exists())
"
```

Expected:
```
n_years: 1.17...
annualized_return: -0.079...  (approx -7.9%)
sharpe_ratio: -57.6...
beats_random: False
IR skipped: True
backups exist: True
```

- [ ] **Step 3: Verify no valid strategies were accidentally hidden**

```bash
python -c "
import json, glob
all_ret = []
for f in glob.glob('artefacts/scalp_research/*/scalp_research_log.json'):
    d = json.load(open(f))
    for r in (d if isinstance(d, list) else d.get('runs',[])):
        m = r.get('test_metrics') or {}
        if m.get('annualized_return'):
            all_ret.append(m['annualized_return'])
print('Max annualized_return:', max(all_ret) if all_ret else 'n/a')
print('Count with ret > 0:', sum(1 for r in all_ret if r > 0))
print('Count total with metrics:', len(all_ret))
"
```

Expected: Metrics are now honest. All strategies were already losing money — max return may still be negative but at a realistic magnitude.

- [ ] **Step 4: Commit the patched logs**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git add code3.0/artefacts/scalp_research/
git commit -m "data: backfill scalp_research logs with corrected 1M metrics

Applied scale=60 correction to all 185 log entries. n_years corrected
from ~70 to ~1.2, annualized returns and Sharpe now reflect actual
performance. Originals backed up as *_backup_prebackfill.json.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Final regression check

- [ ] **Step 1: Run the full test suite one more time**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0/code3.0
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass including the two new test files.

- [ ] **Step 2: Push to GitHub**

```bash
cd C:/Users/LENOVO/Desktop/trade2.0
git push
```
