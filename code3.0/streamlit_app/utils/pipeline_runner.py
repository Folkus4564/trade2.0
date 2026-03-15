"""
utils/pipeline_runner.py - Cached pipeline step wrappers for Streamlit.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# Ensure trade2 package is on path
_ROOT = Path(__file__).parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@st.cache_data(show_spinner="Loading data...")
def load_data(config_hash: str, _config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and split 1H and 5M data. Cached by config hash."""
    from trade2.data.splits import load_split_tf
    train_1h, val_1h, test_1h = load_split_tf("1H", _config)
    train_5m, val_5m, test_5m = load_split_tf("5M", _config)
    return {
        "1H": {"train": train_1h, "val": val_1h, "test": test_1h},
        "5M": {"train": train_5m, "val": val_5m, "test": test_5m},
    }


@st.cache_resource(show_spinner="Loading HMM model...")
def load_hmm_model(_config: Dict[str, Any], artefacts_dir: str):
    """Load pre-trained HMM model. Cached as resource."""
    from trade2.models.hmm import TradeHMM
    model_path = Path(artefacts_dir) / "hmm_regime_model.pkl"
    if not model_path.exists():
        return None
    hmm = TradeHMM(_config)
    hmm.load(model_path)
    return hmm


def build_features(df_1h: pd.DataFrame, df_5m: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build 1H and 5M features (not cached -- re-runs on param change)."""
    from trade2.features.builder import add_1h_features, add_5m_features
    feat_1h = add_1h_features(df_1h, config)
    feat_5m = add_5m_features(df_5m, config)
    return feat_1h, feat_5m


@st.cache_data(show_spinner="Computing regime labels...")
def get_regime(config_hash: str, _hmm, _feat_1h: pd.DataFrame):
    """Compute HMM regime labels and probabilities."""
    from trade2.features.hmm_features import get_hmm_feature_matrix
    X, valid_idx = get_hmm_feature_matrix(_feat_1h, _hmm.feature_cols)
    labels, probs = _hmm.predict(X)
    named = _hmm.name_states(labels, probs, _feat_1h.iloc[valid_idx])
    return named, probs, valid_idx


def generate_and_backtest(
    feat_1h: pd.DataFrame,
    feat_5m: pd.DataFrame,
    hmm,
    config: Dict[str, Any],
    split_name: str,
) -> Dict[str, Any]:
    """Full pipeline: regime -> forward-fill -> signals -> stops -> backtest."""
    from trade2.signals.regime import forward_fill_1h_regime
    from trade2.signals.router import route_signals
    from trade2.signals.generator import compute_stops_regime_aware
    from trade2.backtesting.engine import run_backtest
    from trade2.backtesting.metrics import compute_metrics
    from trade2.features.hmm_features import get_hmm_feature_matrix

    # Regime prediction
    X, valid_idx = get_hmm_feature_matrix(feat_1h, hmm.feature_cols)
    labels_arr, probs = hmm.predict(X)
    named = hmm.name_states(labels_arr, probs, feat_1h.iloc[valid_idx])

    # Forward-fill regime to 5M
    bull_prob = probs[:, hmm.bull_state] if hasattr(hmm, "bull_state") else probs[:, 0]
    bear_prob = probs[:, hmm.bear_state] if hasattr(hmm, "bear_state") else probs[:, 1]

    df_5m_regime = forward_fill_1h_regime(
        feat_5m,
        hmm_labels=named,
        hmm_bull_prob=bull_prob,
        hmm_bear_prob=bear_prob,
        hmm_index=feat_1h.iloc[valid_idx].index,
    )

    # Route signals
    sig_df = route_signals(df_5m_regime, config)

    # Compute stops
    sig_df = compute_stops_regime_aware(sig_df, config)

    # Run backtest
    results = run_backtest(sig_df, config)
    metrics = compute_metrics(results, config)

    return {
        "signals": sig_df,
        "results": results,
        "metrics": metrics,
        "split": split_name,
    }
