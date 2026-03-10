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

# ---- Consistent split boundaries across ALL timeframes (module-level defaults) ----
TRAIN_END = "2022-12-31 23:59"
VAL_END   = "2023-12-31 23:59"
# Test = 2024-01-01 to 2025-06-30

# Known raw CSV paths (module-level defaults)
RAW_PATHS = {
    "1H": RAW_DIR / "XAUUSD_1H_2019_2025.csv",
    "5M": RAW_DIR / "XAUUSD_5M_2019_2025.csv",
}


def _resolve_paths_from_config(config: dict) -> dict:
    """Build RAW_PATHS dict from config, falling back to module-level defaults."""
    resolved = dict(RAW_PATHS)
    if config and "data" in config:
        d = config["data"]
        if "raw_1h_csv" in d:
            resolved["1H"] = ROOT / d["raw_1h_csv"]
        if "raw_5m_csv" in d:
            resolved["5M"] = ROOT / d["raw_5m_csv"]
    return resolved


def _resolve_splits_from_config(config: dict) -> tuple:
    """Return (train_end, val_end) strings from config, falling back to module-level defaults."""
    train_end = TRAIN_END
    val_end   = VAL_END
    if config and "splits" in config:
        s = config["splits"]
        if "train_end" in s:
            train_end = s["train_end"] + " 23:59"
        if "val_end" in s:
            val_end = s["val_end"] + " 23:59"
    return train_end, val_end


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


def _find_raw_csv(timeframe: str, raw_paths: dict = None) -> Path:
    """Find the raw CSV for a given timeframe. Auto-merges yearly files if needed."""
    paths = raw_paths if raw_paths is not None else RAW_PATHS
    # 1. Check primary known path
    if timeframe in paths and paths[timeframe] and paths[timeframe].exists():
        return paths[timeframe]

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


def split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into (train, val, test) by fixed date boundaries.
    Uses the SAME boundaries regardless of timeframe to prevent data leakage.

    Train: 2019-01-01 to 2022-12-31
    Val:   2023-01-01 to 2023-12-31
    Test:  2024-01-01 onwards
    """
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        te = pd.Timestamp(train_end, tz="UTC")
        ve = pd.Timestamp(val_end, tz="UTC")
    else:
        te = pd.Timestamp(train_end)
        ve = pd.Timestamp(val_end)

    train = df[df.index <= te].copy()
    val   = df[(df.index > te) & (df.index <= ve)].copy()
    test  = df[df.index > ve].copy()
    return train, val, test


def load_split_tf(
    timeframe: str,
    config: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data for a given timeframe.
    Automatically finds the right CSV.

    Args:
        timeframe: "1H", "5M", etc.
        config:    Optional config dict. When provided, data paths and split dates
                   are read from config instead of module-level defaults.

    Returns:
        (train, val, test) DataFrames
    """
    raw_paths = _resolve_paths_from_config(config)
    train_end, val_end = _resolve_splits_from_config(config)
    path = _find_raw_csv(timeframe, raw_paths=raw_paths)
    print(f"[loader] Loading {timeframe} data from {path.name}")
    df = load_raw(path)
    train, val, test = split(df, train_end=train_end, val_end=val_end)
    print(f"[loader] {timeframe}: train={len(train)} | val={len(val)} | test={len(test)} bars")
    return train, val, test


def load_multi_tf(config: dict = None) -> dict:
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
        train, val, test = load_split_tf(tf, config=config)
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
