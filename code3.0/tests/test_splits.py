"""
test_splits.py - Tests for train/val/test splitting.
"""

import pandas as pd
import pytest

from trade2.data.splits import split_by_dates


def test_split_boundaries_exact(ohlcv_1h):
    train, val, test = split_by_dates(ohlcv_1h, "2020-06-30", "2020-12-31")
    if len(train) > 0 and len(val) > 0:
        assert train.index[-1] < val.index[0]
    if len(val) > 0 and len(test) > 0:
        assert val.index[-1] < test.index[0]


def test_split_no_overlap(ohlcv_1h):
    train, val, test = split_by_dates(ohlcv_1h, "2020-06-30", "2020-12-31")
    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert len(train_idx & val_idx)  == 0, "Train/val overlap"
    assert len(train_idx & test_idx) == 0, "Train/test overlap"
    assert len(val_idx   & test_idx) == 0, "Val/test overlap"


def test_split_covers_all_data(ohlcv_1h):
    train, val, test = split_by_dates(ohlcv_1h, "2020-06-30", "2020-12-31")
    total_split = len(train) + len(val) + len(test)
    assert total_split == len(ohlcv_1h), f"Split total {total_split} != original {len(ohlcv_1h)}"


def test_split_timezone_aware(ohlcv_1h):
    """Test with timezone-aware index (UTC)."""
    assert ohlcv_1h.index.tz is not None
    train, val, test = split_by_dates(ohlcv_1h, "2020-06-30", "2020-12-31")
    if len(train) > 0:
        assert train.index.tz is not None


def test_split_consistent_across_timeframes(ohlcv_1h, ohlcv_5m):
    """1H and 5M splits should have consistent date boundaries."""
    train_end = "2020-06-30"
    val_end   = "2020-12-31"

    train_1h, val_1h, test_1h = split_by_dates(ohlcv_1h, train_end, val_end)
    train_5m, val_5m, test_5m = split_by_dates(ohlcv_5m, train_end, val_end)

    # 5M should have ~12x more bars than 1H for same period
    if len(train_1h) > 0 and len(train_5m) > 0:
        ratio = len(train_5m) / len(train_1h)
        assert 8 <= ratio <= 16, f"5M/1H ratio {ratio:.1f} outside expected range [8,16]"
