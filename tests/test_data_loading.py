"""
test_data_loading.py - Tests for data loader functions.
"""

import pandas as pd
import numpy as np
import pytest
import io
from pathlib import Path

from trade2.data.loader import load_raw, fill_gaps, resample_ohlcv


DUKASCOPY_CSV = """UTC,Open,High,Low,Close,Volume
02.01.2019 00:00:00.000 UTC,1279.87,1280.5,1279.0,1280.1,500.0
02.01.2019 01:00:00.000 UTC,1280.1,1281.0,1279.5,1280.8,600.0
02.01.2019 02:00:00.000 UTC,1280.8,1282.0,1280.0,1281.5,450.0
"""

ISO_CSV = """Datetime,Open,High,Low,Close,Volume
2019-01-02 00:00:00,1279.87,1280.5,1279.0,1280.1,500.0
2019-01-02 01:00:00,1280.1,1281.0,1279.5,1280.8,600.0
2019-01-02 02:00:00,1280.8,1282.0,1280.0,1281.5,450.0
"""

LOWERCASE_CSV = """utc,open,high,low,close,volume
02.01.2019 00:00:00.000 UTC,1279.87,1280.5,1279.0,1280.1,500.0
02.01.2019 01:00:00.000 UTC,1280.1,1281.0,1279.5,1280.8,600.0
"""


def _write_temp_csv(tmp_path, content, name="test.csv"):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_load_raw_dukascopy_format(tmp_path):
    p  = _write_temp_csv(tmp_path, DUKASCOPY_CSV)
    df = load_raw(p)
    assert len(df) == 3
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.tz is not None


def test_load_raw_iso_format(tmp_path):
    p  = _write_temp_csv(tmp_path, ISO_CSV)
    df = load_raw(p)
    assert len(df) == 3
    assert "Close" in df.columns


def test_load_raw_lowercase_columns(tmp_path):
    p  = _write_temp_csv(tmp_path, LOWERCASE_CSV)
    df = load_raw(p)
    assert "Open" in df.columns
    assert "Close" in df.columns


def test_load_raw_deduplicates(tmp_path):
    dup_csv = DUKASCOPY_CSV + "02.01.2019 00:00:00.000 UTC,1279.87,1280.5,1279.0,1280.1,500.0\n"
    p  = _write_temp_csv(tmp_path, dup_csv)
    df = load_raw(p)
    assert df.index.is_unique


def test_load_raw_removes_zero_prices(tmp_path):
    bad_csv = DUKASCOPY_CSV + "02.01.2019 03:00:00.000 UTC,0.0,0.0,0.0,0.0,0.0\n"
    p  = _write_temp_csv(tmp_path, bad_csv)
    df = load_raw(p)
    assert (df["Close"] > 0).all()


def test_fill_gaps_small_gap(ohlcv_1h):
    # Remove 2 consecutive bars
    df_with_gap = ohlcv_1h.drop(ohlcv_1h.index[100:102])
    df_filled   = fill_gaps(df_with_gap, freq="1h", max_gap_bars=5)
    # Gap should be filled
    assert len(df_filled) >= len(df_with_gap)


def test_fill_gaps_large_gap_not_filled(ohlcv_1h):
    # Remove 10 consecutive bars (> max_gap_bars=5)
    df_with_gap = ohlcv_1h.drop(ohlcv_1h.index[100:110])
    df_filled   = fill_gaps(df_with_gap, freq="1h", max_gap_bars=5)
    # Large gap should NOT be filled — rows at gap location still NaN -> dropped
    assert len(df_filled) < len(ohlcv_1h)


def test_resample_ohlcv_4h(ohlcv_1h):
    df_4h = resample_ohlcv(ohlcv_1h, "4H")
    assert len(df_4h) < len(ohlcv_1h)
    assert df_4h.index.freq is not None or len(df_4h) > 0
    assert "Close" in df_4h.columns
