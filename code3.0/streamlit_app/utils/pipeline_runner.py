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
    from trade2.models.hmm import XAUUSDRegimeModel
    model_path = Path(artefacts_dir) / "models" / "hmm_regime_model.pkl"
    if not model_path.exists():
        return None
    return XAUUSDRegimeModel.load(model_path)


def build_features(df_1h: pd.DataFrame, df_5m: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build 1H and 5M features (not cached -- re-runs on param change)."""
    from trade2.features.builder import add_1h_features, add_5m_features
    feat_1h = add_1h_features(df_1h, config)
    feat_5m = add_5m_features(df_5m, config)
    return feat_1h, feat_5m


@st.cache_data(show_spinner="Computing regime labels...")
def get_regime(config_hash: str, _hmm, _feat_1h: pd.DataFrame, _config: Dict[str, Any]):
    """Compute HMM regime labels and probabilities."""
    from trade2.features.hmm_features import get_hmm_feature_matrix
    X, valid_idx = get_hmm_feature_matrix(_feat_1h, _config)
    labels = _hmm.predict(X)
    probs = _hmm.predict_proba(X)
    named = _hmm.regime_labels(X)
    named.index = _feat_1h.loc[valid_idx].index
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
    from trade2.features.hmm_features import get_hmm_feature_matrix

    # Regime prediction
    X, valid_idx = get_hmm_feature_matrix(feat_1h, config)
    labels_arr = hmm.predict(X)
    probs = hmm.predict_proba(X)
    named = hmm.regime_labels(X)
    named.index = feat_1h.loc[valid_idx].index

    # Forward-fill regime to 5M
    bull_state = hmm.state_map.get("bull", 0)
    bear_state = hmm.state_map.get("bear", 1)
    bull_prob = probs[:, bull_state]
    bear_prob = probs[:, bear_state]

    df_5m_regime = forward_fill_1h_regime(
        feat_5m,
        hmm_labels=named,
        hmm_bull_prob=bull_prob,
        hmm_bear_prob=bear_prob,
        hmm_index=feat_1h.loc[valid_idx].index,
    )

    # Route signals — legacy vs regime_specialized
    strategy_mode = config.get("strategies", {}).get("mode", "legacy")
    if strategy_mode == "regime_specialized":
        sig_df = route_signals(df_5m_regime, config)
        sig_df = compute_stops_regime_aware(sig_df, config)
    else:
        from trade2.signals.generator import generate_signals, compute_stops
        sig_df = generate_signals(df_5m_regime, config)
        p = config["risk"]
        sig_df = compute_stops(sig_df, p["atr_stop_mult"], p["atr_tp_mult"])

    # Run backtest (returns already-computed metrics + trades)
    metrics, trades_df = run_backtest(
        sig_df,
        strategy_name="streamlit_run",
        period_label=split_name,
        config=config,
        freq="5m",
    )

    # Reconstruct equity curve from trades for chart
    init_cash = config["backtest"]["init_cash"]
    if len(trades_df) > 0:
        eq = trades_df.set_index("exit_time")["pnl"].cumsum() + init_cash
        eq.index = pd.to_datetime(eq.index)
        equity_curve = eq.rename("equity").to_frame()
    else:
        equity_curve = None

    return {
        "signals": sig_df,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": metrics,
        "split": split_name,
    }
