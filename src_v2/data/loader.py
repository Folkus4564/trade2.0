"""
Module: loader.py
Purpose: Load and split multi-timeframe XAUUSD data with consistent date boundaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Literal

ROOT     = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---- Consistent split boundaries across ALL timeframes ----
TRAIN_END = "2022-12-31 23:59"
VAL_END   = "2023-12-31 23:59"
# Test = 2024-01-01 to 2025-06-30

# Known raw CSV paths
RAW_PATHS = {
    "1H": RAW_DIR / "XAUUSD_1H_2019_2025.csv",
    "5M": RAW_DIR / "XAUUSD_5M_2019_2025.csv",
}


def _merge_yearly_csvs(timeframe: str) -> Path:
    """
    Merge multiple yearly CSVs (e.g. XAUUSD_5M_2019-2022.csv, XAUUSD_5M_2023-2023.csv,
    XAUUSD_5M_2024-2024.csv) into a single combined file.
    Returns path to merged file, or None if no yearly files found.
    """
    import re
    pattern = re.compile(rf"XAUUSD_{timeframe}_\d{{4}}-\d{{4}}\.csv", re.IGNORECASE)
    yearly_files = sorted(
        [p for p in RAW_DIR.glob(f"XAUUSD_{timeframe}_*.csv") if pattern.match(p.name)],
        key=lambda p: p.name,
    )
    if not yearly_files:
        return None

    out_path = RAW_DIR / f"XAUUSD_{timeframe}_merged.csv"
    # Only re-merge if any source is newer than the merged file
    if out_path.exists():
        merged_mtime = out_path.stat().st_mtime
        if all(p.stat().st_mtime <= merged_mtime for p in yearly_files):
            return out_path

    print(f"[loader] Merging {len(yearly_files)} {timeframe} CSV files...")
    dfs = []
    for p in yearly_files:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  + {p.name}: {len(df)} rows")
        except Exception as e:
            print(f"  ! Skip {p.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate on timestamp column
    ts_col = None
    for c in ["UTC", "time", "Datetime", "datetime"]:
        if c in combined.columns:
            ts_col = c
            break
    if ts_col:
        combined = combined.drop_duplicates(subset=[ts_col]).sort_values(ts_col)
    combined.to_csv(out_path, index=False)
    print(f"[loader] Merged -> {out_path.name} ({len(combined)} rows)")
    return out_path


def _find_raw_csv(timeframe: str) -> Path:
    """Find the raw CSV for a given timeframe. Auto-merges yearly files if needed."""
    # 1. Check primary known path
    if timeframe in RAW_PATHS and RAW_PATHS[timeframe] and RAW_PATHS[timeframe].exists():
        return RAW_PATHS[timeframe]

    # 2. Try merging yearly files (e.g. downloaded per-year batches)
    merged = _merge_yearly_csvs(timeframe)
    if merged and merged.exists():
        return merged

    # 3. Fallback: search raw/ folder for any matching file
    patterns = [
        f"XAUUSD*{timeframe}*.csv",
        f"XAUUSD*{timeframe.lower()}*.csv",
    ]
    for pattern in patterns:
        candidates = sorted(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No raw CSV found for timeframe {timeframe}. "
        f"Run: python src_v2/data/dukascopy_downloader.py"
    )


def load_raw(path: Path) -> pd.DataFrame:
    """
    Load raw Dukascopy CSV. Returns clean OHLCV DataFrame with DatetimeIndex (UTC).
    Handles both old and new Dukascopy formats.
    """
    df = pd.read_csv(path)

    # Dukascopy uses 'UTC' or 'time' or first column as timestamp
    ts_col = None
    for candidate in ["UTC", "time", "Datetime", "datetime", "timestamp"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df.rename(columns={ts_col: "Datetime"}, inplace=True)

    # Try multiple date formats
    for fmt in [
        "%d.%m.%Y %H:%M:%S.%f UTC",
        "%d.%m.%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        None,  # fallback to infer
    ]:
        try:
            if fmt:
                df["Datetime"] = pd.to_datetime(df["Datetime"], format=fmt, utc=True)
            else:
                df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
            break
        except (ValueError, TypeError):
            continue

    df.set_index("Datetime", inplace=True)

    # Normalize column names (duka sometimes uses lowercase)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "open":
            col_map[col] = "Open"
        elif cl == "high":
            col_map[col] = "High"
        elif cl == "low":
            col_map[col] = "Low"
        elif cl == "close":
            col_map[col] = "Close"
        elif cl in ("volume", "vol"):
            col_map[col] = "Volume"
    df.rename(columns=col_map, inplace=True)

    required = {"Open", "High", "Low", "Close"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing OHLCV columns: {missing}. Available: {list(df.columns)}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]

    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into (train, val, test) by fixed date boundaries.
    Uses the SAME boundaries regardless of timeframe to prevent data leakage.

    Train: 2019-01-01 to 2022-12-31
    Val:   2023-01-01 to 2023-12-31
    Test:  2024-01-01 onwards
    """
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        te = pd.Timestamp(TRAIN_END, tz="UTC")
        ve = pd.Timestamp(VAL_END, tz="UTC")
    else:
        te = pd.Timestamp(TRAIN_END)
        ve = pd.Timestamp(VAL_END)

    train = df[df.index <= te].copy()
    val   = df[(df.index > te) & (df.index <= ve)].copy()
    test  = df[df.index > ve].copy()
    return train, val, test


def load_split_tf(timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data for a given timeframe.
    Automatically finds the right CSV.

    Args:
        timeframe: "1H", "5M", etc.

    Returns:
        (train, val, test) DataFrames
    """
    path = _find_raw_csv(timeframe)
    print(f"[loader] Loading {timeframe} data from {path.name}")
    df = load_raw(path)
    train, val, test = split(df)
    print(f"[loader] {timeframe}: train={len(train)} | val={len(val)} | test={len(test)} bars")
    return train, val, test


def load_multi_tf() -> dict:
    """
    Load both 1H and 5M data with consistent splits.

    Returns:
        {
            "1H": {"train": df, "val": df, "test": df},
            "5M": {"train": df, "val": df, "test": df},
        }
    """
    result = {}
    for tf in ["1H", "5M"]:
        train, val, test = load_split_tf(tf)
        result[tf] = {"train": train, "val": val, "test": test}
    return result


if __name__ == "__main__":
    data = load_multi_tf()
    for tf, splits in data.items():
        train = splits["train"]
        test = splits["test"]
        print(f"\n{tf}:")
        print(f"  Train: {train.index[0]} -> {train.index[-1]} ({len(train)} bars)")
        print(f"  Test : {test.index[0]} -> {test.index[-1]} ({len(test)} bars)")
