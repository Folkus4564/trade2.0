"""
data/validation.py - Gap auditing and dataset versioning utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional


def audit_gaps(df: pd.DataFrame, freq: str = "1h") -> Dict[str, Any]:
    """
    Count missing bars vs a complete grid. Returns summary dict.
    """
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, tz=df.index.tz)
    missing = full_index.difference(df.index)
    n_missing = len(missing)

    max_consec = 0
    if n_missing > 0:
        gaps = pd.Series(1, index=missing)
        run_ids = (gaps.index.to_series().diff() != pd.Timedelta(freq)).cumsum()
        max_consec = int(run_ids.value_counts().max())

    return {
        "expected_bars":       len(full_index),
        "actual_bars":         len(df),
        "missing_bars":        n_missing,
        "missing_pct":         round(n_missing / len(full_index) * 100, 2),
        "max_consecutive_gap": max_consec,
    }


def audit_missing_bars(
    df: pd.DataFrame,
    freq: str = "1h",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wrapper around audit_gaps that cross-checks against expected bar count from config.
    """
    gap_info = audit_gaps(df, freq)

    if config and freq == "1h":
        expected = config.get("data", {}).get("expected_bar_count_1h")
        if expected:
            gap_info["expected_from_config"] = expected
            gap_info["config_mismatch"] = abs(len(df) - expected) > expected * 0.05

    pct = gap_info["missing_pct"]
    if pct > 5:
        print(f"[validation] WARNING: {pct:.1f}% bars missing ({gap_info['missing_bars']} bars)")
    return gap_info
