"""
Parallel Dukascopy downloader.
Splits 2019-2024 into 10 date chunks and spawns one subprocess per chunk.
Each saves its own CSV; at the end all are merged into XAUUSD_5M_2019-2024.csv

Usage:
    python download_parallel.py
    python download_parallel.py --workers 80   # workers per process (default 80)
    python download_parallel.py --merge-only   # just merge existing chunk files
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 19 date chunks covering 2019-01-01 to 2024-12-31
# (last original chunk 2023-07-01 to 2024-12-31 split into 10 sub-chunks)
CHUNKS = [
    ("2019-01-01", "2019-06-30"),
    ("2019-07-01", "2019-12-31"),
    ("2020-01-01", "2020-06-30"),
    ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    # --- last chunk split into 10 sub-chunks ---
    ("2023-07-01", "2023-08-24"),
    ("2023-08-25", "2023-10-18"),
    ("2023-10-19", "2023-12-12"),
    ("2023-12-13", "2024-02-05"),
    ("2024-02-06", "2024-04-01"),
    ("2024-04-02", "2024-05-26"),
    ("2024-05-27", "2024-07-20"),
    ("2024-07-21", "2024-09-13"),
    ("2024-09-14", "2024-11-07"),
    ("2024-11-08", "2024-12-31"),
]

DOWNLOADER = ROOT / "src_v2" / "data" / "dukascopy_downloader.py"


def chunk_output(start: str, end: str) -> Path:
    s = start.replace("-", "")
    e = end.replace("-", "")
    return RAW_DIR / f"XAUUSD_5M_chunk_{s}_{e}.csv"


def run_downloads(workers: int):
    procs = []
    for start, end in CHUNKS:
        out = chunk_output(start, end)
        if out.exists():
            print(f"[skip] chunk {start} -> {end} already exists ({out.name})")
            continue

        cmd = [
            sys.executable, str(DOWNLOADER),
            "--start", start,
            "--end",   end,
            "--workers", str(workers),
            "--out",   str(out),
        ]
        log_path = RAW_DIR / f"log_{start[:7]}_{end[:7]}.txt"
        log_file = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        procs.append((start, end, p, log_path))
        print(f"[start] {start} -> {end}  PID={p.pid}  log={log_path.name}")

    if not procs:
        print("[info] All chunks already downloaded. Skipping to merge.")
        return

    print(f"\n[info] {len(procs)} processes running. Waiting for completion...\n")
    t0 = time.time()

    while True:
        alive = [(s, e, p, l) for s, e, p, l in procs if p.poll() is None]
        done  = [(s, e, p, l) for s, e, p, l in procs if p.poll() is not None]
        elapsed = (time.time() - t0) / 60
        print(f"  [{elapsed:.1f}m] {len(done)}/{len(procs)} done  |  "
              f"running: {[s for s,e,p,l in alive]}")

        if not alive:
            break
        time.sleep(30)

    failed = [(s, e) for s, e, p, l in procs if p.returncode != 0]
    if failed:
        print(f"\n[warn] {len(failed)} chunks FAILED: {failed}")
        print("Check the log files in data/raw/ for details.")
    else:
        print(f"\n[ok] All chunks completed in {(time.time()-t0)/60:.1f} min")


def merge_chunks():
    files = sorted(RAW_DIR.glob("XAUUSD_5M_chunk_*.csv"))
    if not files:
        print("[error] No chunk CSVs found in data/raw/")
        return

    print(f"\n[merge] Found {len(files)} chunk files. Merging...")
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        dfs.append(df)
        print(f"  {f.name}: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")

    merged = pd.concat(dfs).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    merged.index.name = "UTC"

    out = RAW_DIR / "XAUUSD_5M_2019-2024.csv"
    merged.to_csv(out)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[merge] Saved {len(merged)} bars -> {out.name} ({size_mb:.1f} MB)")
    print(f"  Range: {merged.index[0]} -> {merged.index[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int, default=80,
                        help="Worker threads per chunk process (default 80)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip downloading, just merge existing chunk CSVs")
    args = parser.parse_args()

    if not args.merge_only:
        run_downloads(args.workers)

    merge_chunks()
