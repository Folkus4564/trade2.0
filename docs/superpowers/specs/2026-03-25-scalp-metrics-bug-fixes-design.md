# Design: Scalp Research Metrics Bug Fixes

**Date:** 2026-03-25
**Status:** Approved

---

## Problem

Two bugs corrupt all scalp research backtest metrics for 1M signal timeframe runs:

1. **`n_years` wrong by 60x** — `_TF_SCALE` in `engine.py` is missing a `"1min"` key, so 1M bars are treated as 1H bars. `bars_per_year = 252*24 = 6048` instead of `252*24*60 = 362,880`. Every derived stat (annualized return, Sharpe, Sortino, ann_vol) is annualized against the wrong horizon.

2. **`signal_source` mode always fails** — `scalp_research_loop.py` constructs `base_model_path` as `artefacts/models/{base_model_id}.pkl` (e.g. `batch_05.pkl`), but `run_pipeline.py` saves models as `hmm_{tf_key}_{n_states}states_{model_id}.pkl` (e.g. `hmm_15m_3states_batch_05.pkl`). Path mismatch causes every `signal_source` mode run to error.

185 existing log entries have wrong metrics and need backfilling.

---

## Approach

Mathematical backfill — fix the source bugs, then apply closed-form scaling to the 185 stored log entries without re-running any backtests.

---

## Section 1 — Engine fix

**File:** `code3.0/src/trade2/backtesting/engine.py`

**Change 1:** Add `"1min"` entries to `_TF_SCALE` at line ~296.

```python
_TF_SCALE = {
    "1min": 60, "1m": 60, "1M": 60,   # 1-minute: 60x more bars than 1H
    "5min": 12, "5m": 12, "5M": 12,
    "15min": 4, "15m": 4, "15M": 4,
    "30min": 2, "30m": 2, "30M": 2,
    "1h": 1, "1H": 1, "4h": 0,
}
```

`bars_per_year` becomes `252 * 24 * 60 = 362,880` for 1M runs. All downstream metrics (n_years, annualized_return, Sharpe, Sortino, ann_vol, calmar, random baseline) auto-correct for all future runs.

**Change 2:** `engine.py` also has a second dict `_TF_TO_FREQ_WF` inside `run_walk_forward()` (line ~455) which is also missing `"1M"`. Add it there too:

```python
_TF_TO_FREQ_WF = {
    "1M": "1min",   # ← add
    "5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"
}
```

Without this, walk-forward runs with `signal_tf="1M"` fall back to `"5min"` (tf_scale=12 instead of 60), giving still-wrong metrics. Note: `run_pipeline.py`'s `_TF_TO_FREQ` dict (line ~236) already contains `"1M": "1min"` and does not need changing.

---

## Section 2 — Model path fix

**File:** `code3.0/src/trade2/app/scalp_research_loop.py`
**Change:** Replace line ~1117's hardcoded path with one derived from config, matching `run_pipeline.py`'s naming convention.

```python
_regime_tf   = base_cfg_clean.get("strategy", {}).get("regime_timeframe", "15M").lower()
_n_states    = base_cfg_clean.get("hmm", {}).get("n_states", 3)
_model_fname = f"hmm_{_regime_tf}_{_n_states}states_{args.base_model_id}.pkl"
base_model_path = str(PROJECT_ROOT / "artefacts" / "models" / _model_fname)
```

This replaces `artefacts/models/batch_05.pkl` with `artefacts/models/hmm_15m_3states_batch_05.pkl`, matching what `run_pipeline.py` actually saves.

---

## Section 3 — Backfill script

**File:** `code3.0/backfill_scalp_metrics.py` (new one-off script, not installed as CLI)

### Scope

Patches every `scalp_research_log.json` in `artefacts/scalp_research/*/` plus the top-level `scalp_research_best.json`.

Only runs with a real `test_metrics.total_return` and `test_metrics.n_years > 0` are patched. Runs with status `TRANSLATION_FAILED` or `CODE_ERROR` and zero metrics are skipped.

Each entry is also checked for its signal timeframe before applying the `scale=60` correction. The script reads `entry.get("config_override", {}).get("strategy", {}).get("signal_timeframe")` and only applies `scale=60` when the value is `"1M"`. Missing or null values default to the scalp.yaml baseline (`"1M"`). Entries with a different signal TF (e.g. 5M) are skipped to avoid double-patching. In practice all current batches use `signal_tf="1M"` (from scalp.yaml), but the guard prevents incorrect patching if future batches differ.

### Correction factor

```
scale = 60   # bars_per_year_new / bars_per_year_old = 362880 / 6048
```

### Per-metric corrections

| Metric | Formula |
|--------|---------|
| `n_years` | `n_years_old / scale` |
| `annualized_return` | `(1 + total_return)^(scale / n_years_old_bars) - 1` = `(1 + total_return)^(1 / n_years_new) - 1` |
| `sharpe_ratio` | `sharpe_old * sqrt(scale)` |
| `sortino_ratio` | derive `down_std_old = (ann_ret_old - 0.04) / sortino_old`; `down_std_new = down_std_old * sqrt(scale)`; `sortino_new = (ann_ret_new - 0.04) / down_std_new`; guard: skip (leave as-is) if `sortino_old == 0` OR `abs(ann_ret_old - 0.04) < 1e-9` |
| `annualized_volatility` | `ann_vol_old * sqrt(scale)` |
| `calmar_ratio` | `ann_ret_new / abs(max_dd)` (max_dd unchanged) |
| `benchmark_return` | derive `bench_total = (1 + bench_ann_old)^n_years_old - 1`; then `(1 + bench_total)^(1/n_years_new) - 1` |
| `alpha_vs_benchmark` | `ann_ret_new - benchmark_return_new` (recomputed after both change) |
| `information_ratio` | `bench_vol_old` is not stored and cannot be recovered; set to `null` and add `"information_ratio_backfill_skipped": true` flag on the metrics object |
| `random_median_sharpe` | `* sqrt(scale)` |
| `random_p95_sharpe` | `* sqrt(scale)` |
| `random_median_return` | same formula as `annualized_return` |
| `beats_random_baseline` | re-evaluate: `sharpe_new > random_median_sharpe_new` |
| `cost_sensitivity_2x.*` | same corrections applied recursively |

### Objects patched per run entry

- `test_metrics` (primary)
- `baseline_metrics` (if present and non-empty)
- `mode_results[*].test_metrics` (all integration modes)

### Safety

- Backs up each `scalp_research_log.json` to `*_backup_prebackfill.json` before overwriting
- Backs up `scalp_research_best.json` similarly
- Prints a summary table of before/after for each patched run

### Post-patch

After all logs are patched, re-evaluate `is_best` within each batch (highest `annualized_return` wins) and rewrite `scalp_research_best.json` with the globally best entry.

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| `code3.0/src/trade2/backtesting/engine.py` | Modified | Add `"1min"` to `_TF_SCALE` |
| `code3.0/src/trade2/backtesting/engine.py` (2nd change) | Modified | Add `"1M": "1min"` to `_TF_TO_FREQ_WF` inside `run_walk_forward()` |
| `code3.0/src/trade2/app/scalp_research_loop.py` | Modified | Fix `base_model_path` derivation |
| `code3.0/backfill_scalp_metrics.py` | New | One-off backfill script |

---

## Verification

1. Run `python backfill_scalp_metrics.py` — check printed before/after table; verify `n_years` drops from ~70 to ~1.2 for completed runs.
2. Spot-check: for `rsi_extreme_snapback` (batch_09), `total_return=-9.23%`, `n_years_old=70.22`:
   - `n_years_new = 70.22/60 = 1.17`
   - `annualized_return_new = (0.9077)^(1/1.17) - 1 ≈ -7.9%`
   - `sharpe_new = -7.44 * sqrt(60) ≈ -57.6`
   - All metrics should be worse (strategy is genuinely losing, just compressed before)
3. Run one fresh scalp_research run on any seed indicator and confirm `n_years ≈ 1.x` in output.
4. Confirm `signal_source` mode no longer errors (check mode_results in log).
