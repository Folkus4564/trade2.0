"""
backfill_scalp_metrics.py - One-off script to correct 1M-timeframe metric errors.

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
    m = dict(m)  # shallow copy - we'll replace nested dicts below too

    n_years_old = m["n_years"]
    n_years_new = n_years_old / scale
    total_return = m.get("total_return", 0.0)

    # --- primary scalars ---
    ann_ret_old = m.get("annualized_return", 0.0)  # capture before overwriting
    m["n_years"] = n_years_new
    m["annualized_return"] = (1 + total_return) ** (1 / n_years_new) - 1
    m["annualized_volatility"] = m.get("annualized_volatility", 0.0) * sqrt_s
    m["sharpe_ratio"] = m.get("sharpe_ratio", 0.0) * sqrt_s

    # sortino: reconstruct down_std_old then rescale
    sortino_old = m.get("sortino_ratio", 0.0)
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


def process_log_file(log_path: Path) -> tuple:
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


def rebuild_best(all_entries: list) -> dict:
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
