"""
test_signals.py - Tests for signal generation logic.
"""

import numpy as np
import pandas as pd
import pytest

from trade2.features.builder import add_1h_features, add_5m_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime
from trade2.signals.generator import generate_signals, compute_stops


@pytest.fixture(scope="module")
def trained_hmm(ohlcv_1h, base_config):
    df = add_1h_features(ohlcv_1h, base_config)
    X, _ = get_hmm_feature_matrix(df)
    # Use fast settings for tests
    model = XAUUSDRegimeModel(n_states=3, n_iter=10, random_seed=42)
    model.fit(X)
    return model, df, X


def test_signals_only_in_correct_regime(ohlcv_1h, base_config, trained_hmm):
    """Long signals should only appear in bull regime, short in bear."""
    model, df_1h, X = trained_hmm
    _, idx = get_hmm_feature_matrix(df_1h)

    labels    = model.regime_labels(X)
    bull_prob = model.bull_probability(X)
    bear_prob = model.bear_probability(X)

    sig = generate_signals(
        df_1h, base_config,
        hmm_labels=labels, hmm_bull_prob=bull_prob, hmm_bear_prob=bear_prob, hmm_index=idx,
    )

    # Where there's a long signal, regime must be bull
    long_rows = sig[sig["signal_long"] == 1]
    if len(long_rows) > 0:
        assert (long_rows["regime"] == "bull").all(), "Long signals in non-bull regime"

    short_rows = sig[sig["signal_short"] == 1]
    if len(short_rows) > 0:
        assert (short_rows["regime"] == "bear").all(), "Short signals in non-bear regime"


def test_no_simultaneous_long_short(ohlcv_1h, base_config, trained_hmm):
    """No bar should have both long and short signals."""
    model, df_1h, X = trained_hmm
    _, idx = get_hmm_feature_matrix(df_1h)

    labels    = model.regime_labels(X)
    bull_prob = model.bull_probability(X)
    bear_prob = model.bear_probability(X)

    sig = generate_signals(
        df_1h, base_config,
        hmm_labels=labels, hmm_bull_prob=bull_prob, hmm_bear_prob=bear_prob, hmm_index=idx,
    )
    simultaneous = (sig["signal_long"] == 1) & (sig["signal_short"] == 1)
    assert not simultaneous.any(), "Simultaneous long and short signals detected"


def test_compute_stops_present(ohlcv_1h, base_config, trained_hmm):
    model, df_1h, X = trained_hmm
    _, idx = get_hmm_feature_matrix(df_1h)
    labels    = model.regime_labels(X)
    bull_prob = model.bull_probability(X)
    bear_prob = model.bear_probability(X)

    sig = generate_signals(
        df_1h, base_config,
        hmm_labels=labels, hmm_bull_prob=bull_prob, hmm_bear_prob=bear_prob, hmm_index=idx,
    )
    sig = compute_stops(sig, atr_stop_mult=1.5, atr_tp_mult=3.0)
    for col in ["stop_long", "stop_short", "tp_long", "tp_short"]:
        assert col in sig.columns, f"Missing column: {col}"


def test_forward_fill_regime_coverage(ohlcv_5m, ohlcv_1h, base_config, trained_hmm):
    """Every 5M bar should have a regime value after forward-fill."""
    model, df_1h, X = trained_hmm
    _, idx = get_hmm_feature_matrix(df_1h)

    labels    = model.regime_labels(X)
    bull_prob = model.bull_probability(X)
    bear_prob = model.bear_probability(X)

    df_5m_feat = add_5m_features(ohlcv_5m, base_config)
    df_5m_with_regime = forward_fill_1h_regime(df_5m_feat, labels, bull_prob, bear_prob, idx)

    assert "regime" in df_5m_with_regime.columns
    assert "bull_prob" in df_5m_with_regime.columns
    # All bars should have a regime (no NaN after ffill + fillna)
    assert not df_5m_with_regime["regime"].isna().any()
