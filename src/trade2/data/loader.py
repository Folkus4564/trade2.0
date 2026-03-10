"""
data/loader.py - Load raw OHLCV CSVs, fill gaps, resample.
No sys.path hacks. No module-level mkdir.
"""

import re
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


TF_RULES = {
    "1H":    "1h",
    "4H":    "4h",
    "Daily": "1D",
}


def load_raw(path: Path) -> pd.DataFrame:
    """
    Load raw Dukascopy CSV. Returns clean OHLCV DataFrame with DatetimeIndex (UTC).
    Handles both old format ('DD.MM.YYYY HH:MM:SS.mmm UTC') and ISO format.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Detect timestamp column
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
        None,
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

    # Normalize column names (Dukascopy sometimes uses lowercase)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "open":        col_map[col] = "Open"
        elif cl == "high":      col_map[col] = "High"
        elif cl == "low":       col_map[col] = "Low"
        elif cl == "close":     col_map[col] = "Close"
        elif cl in ("volume", "vol"): col_map[col] = "Volume"
    df.rename(columns=col_map, inplace=True)

    required = {"Open", "High", "Low", "Close"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    return df


def fill_gaps(df: pd.DataFrame, freq: str = "1h", max_gap_bars: int = 5) -> pd.DataFrame:
    """
    Forward-fill small gaps (<=max_gap_bars) in OHLCV data.
    Larger gaps (weekends, holidays) are left as NaN and dropped.
    """
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, tz=df.index.tz)
    df_full = df.reindex(full_index)

    df_full["_gap"] = df_full["Close"].isna()
    gap_size = df_full["_gap"].groupby(
        (df_full["_gap"] != df_full["_gap"].shift()).cumsum()
    ).transform("sum")

    fill_mask = df_full["_gap"] & (gap_size <= max_gap_bars)
    for col in ["Open", "High", "Low", "Close"]:
        df_full.loc[fill_mask, col] = df_full[col].ffill()[fill_mask]
    df_full.loc[fill_mask, "Volume"] = 0.0

    df_full.drop(columns=["_gap"], inplace=True)
    df_full.dropna(subset=["Close"], inplace=True)
    return df_full


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe. Uses label='left' to prevent lookahead."""
    rule = TF_RULES.get(timeframe, timeframe)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def _find_raw_csv(timeframe: str, raw_dir: Path) -> Path:
    """Find the raw CSV for a given timeframe. Auto-merges yearly files if needed."""
    known = {
        "1H": raw_dir / "XAUUSD_1H_2019_2025.csv",
        "5M": raw_dir / "XAUUSD_5M_2019_2025.csv",
    }
    if timeframe in known and known[timeframe].exists():
        return known[timeframe]

    # Try merging yearly CSV files
    merged = _merge_yearly_csvs(timeframe, raw_dir)
    if merged and merged.exists():
        return merged

    # Fallback: glob for any matching file
    for pattern in [f"XAUUSD*{timeframe}*.csv", f"XAUUSD*{timeframe.lower()}*.csv"]:
        candidates = sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No raw CSV found for timeframe '{timeframe}' in {raw_dir}. "
        f"Run: python scripts/download_data.py"
    )


def _merge_yearly_csvs(timeframe: str, raw_dir: Path) -> Optional[Path]:
    """Merge multiple yearly CSVs into one combined file."""
    pattern = re.compile(rf"XAUUSD_{timeframe}_\d{{4}}-\d{{4}}\.csv", re.IGNORECASE)
    yearly_files = sorted(
        [p for p in raw_dir.glob(f"XAUUSD_{timeframe}_*.csv") if pattern.match(p.name)],
        key=lambda p: p.name,
    )
    if not yearly_files:
        return None

    out_path = raw_dir / f"XAUUSD_{timeframe}_merged.csv"
    if out_path.exists():
        merged_mtime = out_path.stat().st_mtime
        if all(p.stat().st_mtime <= merged_mtime for p in yearly_files):
            return out_path

    print(f"[loader] Merging {len(yearly_files)} {timeframe} CSV files...")
    dfs = []
    for p in yearly_files:
        try:
            dfs.append(pd.read_csv(p))
            print(f"  + {p.name}: {len(dfs[-1])} rows")
        except Exception as e:
            print(f"  ! Skip {p.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    ts_col = next((c for c in ["UTC", "time", "Datetime", "datetime"] if c in combined.columns), None)
    if ts_col:
        combined = combined.drop_duplicates(subset=[ts_col]).sort_values(ts_col)
    combined.to_csv(out_path, index=False)
    print(f"[loader] Merged -> {out_path.name} ({len(combined)} rows)")
    return out_path


def dataset_version(path: Path) -> Dict[str, Any]:
    """Return SHA256 checksum, size and mtime of a raw data file."""
    path = Path(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "file":       path.name,
        "sha256":     sha256,
        "size_bytes": stat.st_size,
        "mtime":      stat.st_mtime,
    }
