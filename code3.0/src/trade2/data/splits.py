"""
data/splits.py - Train/val/test splitting with consistent date boundaries.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from trade2.data.loader import load_raw, fill_gaps, resample_ohlcv, _find_raw_csv


def split_by_dates(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into (train, val, test) by date boundaries.
    Works with both timezone-aware and timezone-naive indices.

    Args:
        df:        OHLCV DataFrame with DatetimeIndex
        train_end: Last date of train period (e.g. "2022-12-31")
        val_end:   Last date of val period (e.g. "2023-12-31")

    Returns:
        (train, val, test) DataFrames
    """
    train_end_str = train_end + " 23:59"
    val_end_str   = val_end   + " 23:59"

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


def load_split_tf(
    timeframe: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience: find CSV -> load -> optional fill_gaps -> split.

    Args:
        timeframe: "1H", "5M", "4H", etc.
        config:    Config dict (from load_config)

    Returns:
        (train, val, test) DataFrames
    """
    data_cfg   = config.get("data", {})
    splits_cfg = config.get("splits", {})

    # Resolve raw CSV path
    project_root = Path(__file__).parents[4]
    raw_dir = project_root / "data" / "raw"

    # Allow config to override known paths
    if timeframe == "1H" and data_cfg.get("raw_1h_csv"):
        csv_path = project_root / data_cfg["raw_1h_csv"]
        if not csv_path.exists():
            csv_path = _find_raw_csv(timeframe, raw_dir)
    elif timeframe == "5M" and data_cfg.get("raw_5m_csv"):
        csv_path = project_root / data_cfg["raw_5m_csv"]
        if not csv_path.exists():
            csv_path = _find_raw_csv(timeframe, raw_dir)
    else:
        csv_path = _find_raw_csv(timeframe, raw_dir)

    print(f"[splits] Loading {timeframe} from {csv_path.name}")
    df = load_raw(csv_path)

    policy = data_cfg.get("missing_bar_policy", "none")
    if policy == "forward_fill" and timeframe == "1H":
        df = fill_gaps(df, freq="1h")

    train, val, test = split_by_dates(
        df,
        train_end=splits_cfg["train_end"],
        val_end=splits_cfg["val_end"],
    )
    print(f"[splits] {timeframe}: train={len(train)} | val={len(val)} | test={len(test)} bars")
    return train, val, test
