"""
Module: loader.py
Purpose: Load and cache XAUUSD OHLCV data with timeframe resampling and train/val/test splits
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Tuple, Dict, Any

from src.config import get_config

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

# ── Resample rules ────────────────────────────────────────────────────────────
TF_RULES = {
    "1H":    "1h",
    "4H":    "4h",
    "Daily": "1D",
}


def _raw_csv_path() -> Path:
    cfg = get_config()
    return ROOT / cfg["data"]["raw_1h_csv"]


# Module-level alias for backward compatibility (evaluated lazily via function)
def _get_raw_csv() -> Path:
    return _raw_csv_path()


# RAW_CSV for backward compat — other modules import this
RAW_CSV = DATA_DIR / "XAUUSD_1H_2019_2024.csv"  # default fallback; overridden by config at runtime


def audit_gaps(df: pd.DataFrame, freq: str = "1h") -> Dict[str, Any]:
    """
    Count missing bars, max consecutive gap, and return summary dict.
    Compares actual bar count against expected full-grid count.
    """
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, tz=df.index.tz)
    missing = full_index.difference(df.index)
    n_missing = len(missing)

    # Max consecutive gap
    if n_missing > 0:
        # Identify runs of consecutive missing timestamps
        gaps = pd.Series(1, index=missing)
        run_ids = (gaps.index.to_series().diff() != pd.Timedelta(freq)).cumsum()
        max_consec = int(run_ids.value_counts().max())
    else:
        max_consec = 0

    return {
        "expected_bars": len(full_index),
        "actual_bars":   len(df),
        "missing_bars":  n_missing,
        "missing_pct":   round(n_missing / len(full_index) * 100, 2),
        "max_consecutive_gap": max_consec,
    }


def dataset_version(path: Path = None) -> dict:
    """
    Return a version tag for a raw data file.
    Includes SHA256 checksum, file size, and modification time.
    Use this to verify that a model was trained on the exact same dataset.
    """
    if path is None:
        path = _raw_csv_path()

    cfg = get_config()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    stat   = path.stat()
    tag = {
        "file":     path.name,
        "sha256":   sha256,
        "size_bytes": stat.st_size,
        "mtime":    stat.st_mtime,
    }

    # Warn if SHA256 doesn't match expected (Phase F)
    expected_sha = cfg.get("data", {}).get("expected_sha256")
    if expected_sha and sha256 != expected_sha:
        print(f"[loader] WARNING: SHA256 mismatch! Expected {expected_sha[:12]}... got {sha256[:12]}...")
        tag["sha256_mismatch"] = True

    version_path = path.parent / f"{path.stem}_version.json"
    version_path.write_text(json.dumps(tag, indent=2))
    return tag


def audit_missing_bars(df: pd.DataFrame, freq: str = "1h") -> Dict[str, Any]:
    """
    Check expected vs actual bar count. Warns if significant bars are missing.
    """
    cfg = get_config()
    expected_key = f"expected_bar_count_{freq.replace('h','h').replace('1h','1h')}"
    expected_from_cfg = cfg.get("data", {}).get("expected_bar_count_1h") if freq == "1h" else None

    gap_info = audit_gaps(df, freq)

    if expected_from_cfg:
        gap_info["expected_from_config"] = expected_from_cfg
        gap_info["config_mismatch"] = abs(len(df) - expected_from_cfg) > expected_from_cfg * 0.05

    pct = gap_info["missing_pct"]
    if pct > 5:
        print(f"[loader] WARNING: {pct:.1f}% bars missing ({gap_info['missing_bars']} bars)")
    return gap_info


def load_raw(path: Path = None) -> pd.DataFrame:
    """
    Load raw Dukascopy CSV. Returns clean OHLCV DataFrame with DatetimeIndex (UTC).
    Handles Dukascopy format: UTC column with 'DD.MM.YYYY HH:MM:SS.mmm UTC' dates.
    """
    if path is None:
        path = _raw_csv_path()

    df = pd.read_csv(path)
    # Dukascopy uses 'UTC' as timestamp column name
    ts_col = "UTC" if "UTC" in df.columns else df.columns[0]
    df.rename(columns={ts_col: "Datetime"}, inplace=True)

    # Parse Dukascopy date format: "01.01.2019 23:00:00.000 UTC"
    try:
        df["Datetime"] = pd.to_datetime(
            df["Datetime"], format="%d.%m.%Y %H:%M:%S.%f UTC", utc=True
        )
    except Exception:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    df.set_index("Datetime", inplace=True)

    required = {"Open", "High", "Low", "Close"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Basic price sanity check
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    return df


def fill_gaps(df: pd.DataFrame, freq: str = "1h", max_gap_bars: int = 5) -> pd.DataFrame:
    """
    Fill missing bars in an OHLCV DataFrame.

    Policy (from config: missing_bar_policy = forward_fill):
    - Reindex to a complete grid at `freq` frequency
    - For gaps <= max_gap_bars: forward-fill OHLC from the last known bar, volume = 0
    - For gaps > max_gap_bars (e.g., weekends, holidays): leave as NaN (will be dropped by features)
    """
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, tz=df.index.tz)
    df_full    = df.reindex(full_index)

    # Mark which bars were originally missing
    df_full["_gap"] = df_full["Close"].isna()

    # Gap size: count consecutive missing bars
    gap_size = df_full["_gap"].groupby(
        (df_full["_gap"] != df_full["_gap"].shift()).cumsum()
    ).transform("sum")

    # Forward-fill only small gaps
    fill_mask = df_full["_gap"] & (gap_size <= max_gap_bars)
    for col in ["Open", "High", "Low", "Close"]:
        df_full.loc[fill_mask, col] = df_full[col].ffill()[fill_mask]
    df_full.loc[fill_mask, "Volume"] = 0.0

    df_full.drop(columns=["_gap"], inplace=True)
    df_full.dropna(subset=["Close"], inplace=True)
    return df_full


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1H OHLCV DataFrame to a higher timeframe.
    Uses label='left', closed='left' to prevent lookahead.
    """
    rule = TF_RULES.get(timeframe, timeframe)
    agg = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna()
    return out


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into (train, val, test) by config date boundaries.
    Handles both timezone-aware and timezone-naive indices.
    """
    cfg = get_config()
    sp = cfg["splits"]
    train_end_str = sp["train_end"] + " 23:59"
    val_end_str   = sp["val_end"]   + " 23:59"

    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        te = pd.Timestamp(train_end_str, tz="UTC")
        ve = pd.Timestamp(val_end_str,   tz="UTC")
    else:
        te = pd.Timestamp(train_end_str)
        ve = pd.Timestamp(val_end_str)

    train = df[df.index <= te].copy()
    val   = df[(df.index > te) & (df.index <= ve)].copy()
    test  = df[df.index > ve].copy()
    return train, val, test


def load_split(
    timeframe: Literal["1H", "4H", "Daily"] = "1H",
    path: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience: load raw -> fill gaps -> optional resample -> split.
    Returns (train, val, test).
    """
    if path is None:
        path = _raw_csv_path()

    df = load_raw(path)

    # Wire fill_gaps into the load pipeline (Phase C3)
    cfg = get_config()
    policy = cfg.get("data", {}).get("missing_bar_policy", "none")
    if policy == "forward_fill":
        df = fill_gaps(df, freq="1h")

    if timeframe != "1H":
        df = resample_ohlcv(df, timeframe)
    return split(df)


def save_processed(
    timeframe: Literal["1H", "4H", "Daily"] = "1H",
    path: Path = None,
) -> None:
    """Save train/val/test parquet files to data/processed/."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split(timeframe, path)
    prefix = f"XAUUSD_{timeframe}"
    train.to_parquet(PROC_DIR / f"{prefix}_train.parquet")
    val.to_parquet(PROC_DIR   / f"{prefix}_val.parquet")
    test.to_parquet(PROC_DIR  / f"{prefix}_test.parquet")
    print(f"[loader] Saved {timeframe}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    for tf in ["1H", "4H", "Daily"]:
        save_processed(tf)
    train, val, test = load_split("1H")
    print(f"\nTrain : {train.index[0]} -> {train.index[-1]}  ({len(train)} bars)")
    print(f"Val   : {val.index[0]} -> {val.index[-1]}  ({len(val)} bars)")
    print(f"Test  : {test.index[0]} -> {test.index[-1]}  ({len(test)} bars)")
