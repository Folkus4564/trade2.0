"""
Module: loader.py
Purpose: Load and cache XAUUSD OHLCV data with timeframe resampling and train/val/test splits
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV  = DATA_DIR / "XAUUSD_1H_2019_2024.csv"
PROC_DIR = DATA_DIR / "processed"

# ── Split boundaries ──────────────────────────────────────────────────────────
TRAIN_END = "2022-12-31 23:59"
VAL_END   = "2023-12-31 23:59"
# Test = 2024-01-01 onwards

# ── Resample rules ────────────────────────────────────────────────────────────
TF_RULES = {
    "1H":    "1h",
    "4H":    "4h",
    "Daily": "1D",
}


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Load raw Dukascopy CSV. Returns clean OHLCV DataFrame with DatetimeIndex (UTC).
    Handles Dukascopy format: UTC column with 'DD.MM.YYYY HH:MM:SS.mmm UTC' dates.
    """
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
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, infer_datetime_format=True)

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
    Split DataFrame into (train, val, test) by fixed date boundaries.
    Train: 2019-2022 | Val: 2023 | Test: 2024
    Handles both timezone-aware and timezone-naive indices.
    """
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        import pytz
        te = pd.Timestamp(TRAIN_END, tz="UTC")
        ve = pd.Timestamp(VAL_END,   tz="UTC")
    else:
        te = pd.Timestamp(TRAIN_END)
        ve = pd.Timestamp(VAL_END)

    train = df[df.index <= te].copy()
    val   = df[(df.index > te) & (df.index <= ve)].copy()
    test  = df[df.index > ve].copy()
    return train, val, test


def load_split(
    timeframe: Literal["1H", "4H", "Daily"] = "1H",
    path: Path = RAW_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience: load raw -> optional resample -> split.
    Returns (train, val, test).
    """
    df = load_raw(path)
    if timeframe != "1H":
        df = resample_ohlcv(df, timeframe)
    return split(df)


def save_processed(
    timeframe: Literal["1H", "4H", "Daily"] = "1H",
    path: Path = RAW_CSV,
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
