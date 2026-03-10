"""
test_features.py - Tests for feature engineering, with lookahead bias checks.
"""

import numpy as np
import pandas as pd
import pytest

from trade2.features.hmm_features import add_hmm_features, get_hmm_feature_matrix
from trade2.features.smc import add_smc_features
from trade2.features.builder import add_1h_features, add_5m_features


HMM_FEAT_COLS = [
    "hmm_feat_ret", "hmm_feat_rsi", "hmm_feat_atr",
    "hmm_feat_vol", "hmm_feat_hma_slope", "hmm_feat_bb_width", "hmm_feat_macd",
]


def test_hmm_features_present(ohlcv_1h):
    df = add_hmm_features(ohlcv_1h)
    for col in HMM_FEAT_COLS:
        assert col in df.columns, f"Missing HMM feature: {col}"


def test_hmm_features_are_lagged(ohlcv_1h):
    """
    CRITICAL lookahead check:
    hmm_feat_* at index i should NOT correlate with the current bar's close.
    Verify by checking that hmm_feat_ret at index i equals log_ret at index i-1.
    """
    df = add_hmm_features(ohlcv_1h)
    # Skip first 2 rows (NaN from shift)
    df_valid = df.dropna(subset=["hmm_feat_ret", "log_ret"])
    if len(df_valid) < 10:
        pytest.skip("Insufficient rows after dropna")

    # hmm_feat_ret[i] should == log_ret[i-1]
    feat_ret  = df["hmm_feat_ret"].dropna()
    log_ret   = df["log_ret"]

    # Align: feat_ret at position j should equal log_ret at position j-1
    feat_vals = feat_ret.values
    log_vals  = log_ret.reindex(feat_ret.index).values

    # Shift comparison: feat_ret at row i == log_ret.shift(1) at row i
    log_ret_shifted = df["log_ret"].shift(1)
    matches = np.allclose(
        df["hmm_feat_ret"].dropna().values,
        log_ret_shifted.reindex(df["hmm_feat_ret"].dropna().index).values,
        equal_nan=True,
    )
    assert matches, "hmm_feat_ret is NOT correctly lagged by 1 bar — possible lookahead!"


def test_hmm_features_no_future_data(ohlcv_1h):
    """
    Verify that at index i, hmm_feat_* values do NOT use close[i].
    Check: hmm_feat_ret[i] != log(close[i] / close[i-1]).
    The feature should equal log(close[i-1] / close[i-2]).
    """
    df = add_hmm_features(ohlcv_1h)
    close = df["Close"]

    # Current bar return: log(close[i] / close[i-1])
    current_ret = np.log(close / close.shift(1))
    # Feature ret (should be prior bar): should NOT match current bar
    feat_ret = df["hmm_feat_ret"]

    # Check a window of valid rows
    valid_idx = df.dropna(subset=["hmm_feat_ret"]).index[5:50]
    current_at_valid = current_ret.reindex(valid_idx).values
    feat_at_valid    = feat_ret.reindex(valid_idx).values

    # They should NOT be identical (feat should be lagged by 1)
    assert not np.allclose(current_at_valid, feat_at_valid), \
        "hmm_feat_ret matches current bar return — possible lookahead!"


def test_smc_features_present(ohlcv_1h):
    df = add_smc_features(ohlcv_1h)
    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        assert col in df.columns, f"Missing SMC feature: {col}"


def test_smc_features_are_boolean(ohlcv_1h):
    df = add_smc_features(ohlcv_1h)
    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        assert df[col].dtype == bool, f"{col} is not bool"


def test_smc_features_no_nan(ohlcv_1h):
    df = add_smc_features(ohlcv_1h)
    for col in ["ob_bullish", "ob_bearish", "fvg_bullish", "fvg_bearish", "sweep_low", "sweep_high"]:
        assert not df[col].isna().any(), f"{col} has NaN values"


def test_get_hmm_feature_matrix_no_nan(ohlcv_1h):
    df = add_hmm_features(ohlcv_1h)
    X, idx = get_hmm_feature_matrix(df)
    assert not np.isnan(X).any(), "HMM feature matrix contains NaN"
    assert X.shape[1] == 7, f"Expected 7 HMM features, got {X.shape[1]}"


def test_add_1h_features_shape(ohlcv_1h, base_config):
    df = add_1h_features(ohlcv_1h, base_config)
    assert len(df) == len(ohlcv_1h), "Feature engineering changed row count"
    assert df.shape[1] > 10, "Too few feature columns"


def test_add_5m_features_shape(ohlcv_5m, base_config):
    df = add_5m_features(ohlcv_5m, base_config)
    assert len(df) == len(ohlcv_5m), "Feature engineering changed row count"
    for col in ["ob_bullish", "fvg_bullish", "sweep_low", "atr_14", "adx_14"]:
        assert col in df.columns, f"Missing 5M feature: {col}"
