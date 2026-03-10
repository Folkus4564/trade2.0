"""
Module: dukascopy_downloader.py
Purpose: Download XAUUSD tick/OHLCV data from Dukascopy and resample to target timeframe.
         Downloads bi5 (LZMA-compressed binary) files directly, decodes to ticks,
         then resamples to 5M OHLCV bars.
         Uses parallel batch downloads (200 workers) for speed.

Usage:
    python src_v2/data/dukascopy_downloader.py
    python src_v2/data/dukascopy_downloader.py --start 2019-01-01 --end 2024-12-31 --workers 200
"""

import struct
import lzma
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import sys

ROOT    = Path(__file__).parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DUKA_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# XAUUSD tick format: uint32 ms_offset, uint32 ask*1000, uint32 bid*1000, float ask_vol, float bid_vol
TICK_FMT  = ">IIIff"
TICK_SIZE = struct.calcsize(TICK_FMT)  # 20 bytes

# Thread-local storage for per-thread sessions
_local = threading.local()


def _get_session() -> requests.Session:
    """Get or create a thread-local requests session."""
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Encoding": "gzip, deflate, br",
        })
        _local.session = s
    return _local.session


def _is_trading_hour(dt: datetime) -> bool:
    """Return True if this UTC hour is a likely XAUUSD trading hour (Mon-Fri)."""
    # weekday(): Mon=0 .. Sun=6
    # XAUUSD closes ~22:00 UTC Friday, reopens ~22:00 UTC Sunday
    wd = dt.weekday()
    if wd == 5:  # Saturday
        return False
    if wd == 6 and dt.hour < 22:  # Sunday before market open
        return False
    if wd == 4 and dt.hour >= 22:  # Friday after close
        return False
    return True


def _fetch_hour(symbol: str, dt: datetime, retries: int = 5) -> tuple[datetime, bytes]:
    """Fetch one hour of tick data using a thread-local session. Returns (dt, raw_bytes)."""
    url = DUKA_URL.format(
        symbol=symbol,
        year=dt.year,
        month=dt.month - 1,   # Dukascopy months are 0-indexed
        day=dt.day,
        hour=dt.hour,
    )
    session = _get_session()
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 0:
                return dt, r.content
            if r.status_code in (429, 503):
                # Rate limited — back off and retry
                time.sleep(2 ** attempt)
                continue
            # Empty response on a trading hour = likely transient; retry once
            if r.status_code == 200 and len(r.content) == 0:
                if _is_trading_hour(dt) and attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            return dt, b""
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
    return dt, b""


def _decode_bi5(raw: bytes, dt: datetime, point: float) -> pd.DataFrame:
    """Decode bi5 binary tick data into a DataFrame."""
    if not raw:
        return pd.DataFrame()
    try:
        data = lzma.decompress(raw)
    except Exception:
        return pd.DataFrame()

    n = len(data) // TICK_SIZE
    if n == 0:
        return pd.DataFrame()

    base_ts = pd.Timestamp(dt, tz="UTC")
    # Vectorized decode
    arr = np.frombuffer(data[:n * TICK_SIZE], dtype=np.dtype([
        ("ms",      ">u4"),
        ("ask_raw", ">u4"),
        ("bid_raw", ">u4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]))
    ms_offsets = arr["ms"].astype(np.int64)
    mid        = (arr["ask_raw"] + arr["bid_raw"]) * point / 2.0
    vol        = (arr["ask_vol"] + arr["bid_vol"]).astype(float)
    timestamps = base_ts + pd.to_timedelta(ms_offsets, unit="ms")

    df = pd.DataFrame({"Mid": mid, "Volume": vol}, index=timestamps)
    df.index.name = "Datetime"
    return df


def _ticks_to_ohlcv(ticks: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample mid-price ticks to OHLCV bars."""
    if ticks.empty:
        return pd.DataFrame()
    mid = ticks["Mid"]
    vol = ticks["Volume"]
    ohlcv = pd.DataFrame({
        "Open":   mid.resample(freq, label="left", closed="left").first(),
        "High":   mid.resample(freq, label="left", closed="left").max(),
        "Low":    mid.resample(freq, label="left", closed="left").min(),
        "Close":  mid.resample(freq, label="left", closed="left").last(),
        "Volume": vol.resample(freq, label="left", closed="left").sum(),
    }).dropna(subset=["Open"])
    return ohlcv


def download_xauusd_5m(
    start:       date  = date(2019, 1, 1),
    end:         date  = date(2024, 12, 31),
    output_path: Path  = None,
    workers:     int   = 200,
    freq:        str   = "5min",
) -> Path:
    """
    Download XAUUSD tick data from Dukascopy and resample to 5M OHLCV.
    Uses parallel batch downloads for speed (~200 concurrent HTTP requests).

    Args:
        start:       Start date (inclusive)
        end:         End date (inclusive)
        output_path: Where to save the CSV (auto-named if None)
        workers:     Number of parallel download threads (default 200)
        freq:        Resample frequency (default "5min")

    Returns:
        Path to saved CSV
    """
    if output_path is None:
        output_path = RAW_DIR / f"XAUUSD_5M_{start.year}-{end.year}.csv"

    symbol = "XAUUSD"
    point  = 0.001   # XAUUSD: tick data stored as price * 1000

    start_dt = datetime(start.year, start.month, start.day, 0, 0, 0)
    end_dt   = datetime(end.year,   end.month,   end.day,  23, 0, 0)

    # Build list of all hours to download
    hours = []
    cur = start_dt
    while cur <= end_dt:
        hours.append(cur)
        cur += timedelta(hours=1)

    total = len(hours)
    print(f"[downloader] {symbol} {start} -> {end}  |  {total} hours  |  {workers} workers")
    sys.stdout.flush()

    all_bars  = []
    done      = 0
    t0        = time.time()

    # Batch size: submit `workers` requests at a time, then small pause to avoid rate-limit
    BATCH       = workers
    BATCH_PAUSE = 0.3   # seconds between batches (keeps ~workers req/s sustainable)

    for batch_start in range(0, total, BATCH):
        batch = hours[batch_start: batch_start + BATCH]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_hour, symbol, h): h for h in batch}
            for fut in as_completed(futures):
                dt, raw = fut.result()
                ticks   = _decode_bi5(raw, dt, point)
                if not ticks.empty:
                    bars = _ticks_to_ohlcv(ticks, freq)
                    if not bars.empty:
                        all_bars.append(bars)
                done += 1

        time.sleep(BATCH_PAUSE)

        elapsed = time.time() - t0
        pct     = done / total * 100
        eta     = (elapsed / done * (total - done)) if done > 0 else 0
        cur_dt  = batch[-1].strftime("%Y-%m-%d")
        bar_count = sum(len(b) for b in all_bars)
        print(f"  [{pct:5.1f}%] {cur_dt}  |  {done}/{total} hrs  |  "
              f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  |  bars: {bar_count}")
        sys.stdout.flush()

    if not all_bars:
        raise RuntimeError("No data downloaded. Check internet or Dukascopy availability.")

    df = pd.concat(all_bars).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "UTC"

    df.to_csv(output_path)
    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\n[downloader] Done in {elapsed/60:.1f} min")
    print(f"[downloader] Saved {len(df)} bars -> {output_path.name} ({size_mb:.1f} MB)")
    print(f"  Range: {df.index[0]} -> {df.index[-1]}")
    sys.stdout.flush()

    return output_path


def download_years(
    years:   list[int],
    workers: int = 50,
) -> Path:
    """
    Download one year at a time, saving each to its own file.
    Skips years that are already fully downloaded.
    Merges all years into XAUUSD_5M_2019-2024.csv at the end.

    Returns path to merged file.
    """
    yearly_paths = []
    for yr in years:
        out_path = RAW_DIR / f"XAUUSD_5M_{yr}-{yr}.csv"

        # Check if year already downloaded with enough bars
        if out_path.exists():
            try:
                existing = pd.read_csv(out_path)
                # Expect at least 50k bars for a full year (~72k expected)
                if len(existing) >= 50_000:
                    print(f"[downloader] {yr}: already complete ({len(existing)} bars), skipping")
                    yearly_paths.append(out_path)
                    continue
                else:
                    print(f"[downloader] {yr}: incomplete ({len(existing)} bars), re-downloading...")
            except Exception:
                pass

        path = download_xauusd_5m(
            start       = date(yr, 1, 1),
            end         = date(yr, 12, 31),
            output_path = out_path,
            workers     = workers,
        )
        yearly_paths.append(path)

    # Merge all years into one file
    merged_path = RAW_DIR / "XAUUSD_5M_2019-2024.csv"
    print(f"\n[downloader] Merging {len(yearly_paths)} yearly files...")
    dfs = []
    for p in sorted(yearly_paths):
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  + {p.name}: {len(df)} bars")

    combined = pd.concat(dfs, ignore_index=True)
    ts_col = "UTC" if "UTC" in combined.columns else combined.columns[0]
    combined = combined.drop_duplicates(subset=[ts_col]).sort_values(ts_col)
    combined.to_csv(merged_path, index=False)
    size_mb = merged_path.stat().st_size / 1e6
    print(f"[downloader] Merged -> {merged_path.name} ({len(combined)} bars, {size_mb:.1f} MB)")
    return merged_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2019-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--out",     default=None)
    parser.add_argument("--by-year", action="store_true",
                        help="Download year-by-year with resume support (recommended)")
    args = parser.parse_args()

    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)

    if args.by_year:
        years = list(range(s.year, e.year + 1))
        download_years(years, workers=args.workers)
    else:
        out = Path(args.out) if args.out else None
        download_xauusd_5m(start=s, end=e, output_path=out, workers=args.workers)
