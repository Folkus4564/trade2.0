"""
merge_scalp_batches.py

Merges all 20 batch scalp_research logs into a single
artefacts/scalp_research/scalp_research_log.json.

Usage:
    cd code3.0/
    python merge_scalp_batches.py
"""

import json
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent
RESEARCH_DIR    = PROJECT_ROOT / "artefacts" / "scalp_research"
N_BATCHES       = 20
OUT_LOG         = RESEARCH_DIR / "scalp_research_log.json"
OUT_BEST        = RESEARCH_DIR / "scalp_research_best.json"


def main():
    all_entries = []

    for i in range(1, N_BATCHES + 1):
        batch_log = RESEARCH_DIR / f"batch_{i:02d}" / "scalp_research_log.json"
        if not batch_log.exists():
            print(f"[merge] batch_{i:02d}: log not found, skipping")
            continue
        with open(batch_log) as f:
            entries = json.load(f)
        print(f"[merge] batch_{i:02d}: {len(entries)} entries")
        all_entries.extend(entries)

    if not all_entries:
        print("[merge] No entries found across any batch.")
        return

    # Deduplicate by name (keep highest-return entry per indicator)
    by_name = {}
    for e in all_entries:
        name = e.get("name", "")
        if not name:
            continue
        cur_ret = (e.get("test_metrics") or {}).get("annualized_return", -999)
        prev_ret = (by_name.get(name, {}).get("test_metrics") or {}).get("annualized_return", -999)
        if cur_ret > prev_ret:
            by_name[name] = e

    merged = list(by_name.values())

    # Sort by return descending, reassign ids
    merged.sort(key=lambda e: (e.get("test_metrics") or {}).get("annualized_return", -999), reverse=True)
    for i, e in enumerate(merged, 1):
        e["id"] = i

    # Mark overall best
    best_entry = None
    for e in merged:
        ret = (e.get("test_metrics") or {}).get("annualized_return", -999)
        if best_entry is None or ret > (best_entry.get("test_metrics") or {}).get("annualized_return", -999):
            best_entry = e
    if best_entry:
        best_entry["is_best"] = True

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    if best_entry:
        with open(OUT_BEST, "w") as f:
            json.dump(best_entry, f, indent=2, default=str)

    print(f"\n[merge] Done.")
    print(f"[merge] Total entries: {len(all_entries)} -> {len(merged)} after dedup")
    print(f"[merge] Written to: {OUT_LOG}")
    if best_entry:
        ret = (best_entry.get("test_metrics") or {}).get("annualized_return", 0)
        sh  = (best_entry.get("test_metrics") or {}).get("sharpe_ratio", 0)
        print(f"[merge] Best: {best_entry['name']} | return={ret*100:.1f}% | sharpe={sh:.2f}")


if __name__ == "__main__":
    main()
