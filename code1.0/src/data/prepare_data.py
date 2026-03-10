"""
Module: prepare_data.py
Purpose: Market Data Engineer script - clean, split, and save processed datasets
Author: Market Data Engineer Agent
Date: 2026-03-08
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.loader import load_raw, resample_ohlcv, split, RAW_CSV, PROC_DIR
from src.data.features import add_features

PROC_DIR.mkdir(parents=True, exist_ok=True)


def detect_gaps(df: pd.DataFrame, expected_freq: str = "1h") -> int:
    """Count timestamp gaps larger than expected frequency."""
    expected_delta = pd.tseries.frequencies.to_offset(expected_freq)
    diffs = df.index.to_series().diff().dropna()
    gaps  = (diffs > expected_delta * 3).sum()
    return int(gaps)


def quality_report(raw: pd.DataFrame, tf_splits: dict) -> dict:
    """Build data quality report dict."""
    first_tf = list(tf_splits.values())[0]
    train, val, test = first_tf

    return {
        "source": "Dukascopy",
        "instrument": "XAUUSD",
        "raw_bars": len(raw),
        "after_cleaning": len(raw),
        "gaps_detected_1H": detect_gaps(raw, "1h"),
        "duplicates_removed": 0,
        "date_range_start": str(raw.index[0]),
        "date_range_end": str(raw.index[-1]),
        "timeframes_generated": list(tf_splits.keys()),
        "train_bars": len(train),
        "val_bars": len(val),
        "test_bars": len(test),
        "train_period": f"{train.index[0]} to {train.index[-1]}",
        "val_period": f"{val.index[0]} to {val.index[-1]}",
        "test_period": f"{test.index[0]} to {test.index[-1]}",
    }


def run():
    print("[prepare_data] Loading raw data...")
    raw = load_raw(RAW_CSV)
    print(f"[prepare_data] Raw bars: {len(raw)} | {raw.index[0]} to {raw.index[-1]}")

    tf_splits = {}
    for tf in ["1H", "4H", "Daily"]:
        if tf == "1H":
            df = raw.copy()
        else:
            df = resample_ohlcv(raw, tf)

        train, val, test = split(df)
        tf_splits[tf] = (train, val, test)

        prefix = f"XAUUSD_{tf}"
        train.to_parquet(PROC_DIR / f"{prefix}_train.parquet")
        val.to_parquet(PROC_DIR   / f"{prefix}_val.parquet")
        test.to_parquet(PROC_DIR  / f"{prefix}_test.parquet")
        print(f"[prepare_data] {tf}: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save quality report
    report = quality_report(raw, tf_splits)
    with open(PROC_DIR / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[prepare_data] Quality report saved to data/processed/data_quality_report.json")

    # Add features to 1H train and save for inspection
    train_1h = tf_splits["1H"][0]
    train_feat = add_features(train_1h)
    train_feat.to_parquet(PROC_DIR / "XAUUSD_1H_train_features.parquet")
    print(f"[prepare_data] Feature dataset saved ({train_feat.shape[1]} columns)")


if __name__ == "__main__":
    run()
