"""
Module: hmm_model.py
Purpose: HMM regime detection - train, predict, and persist Gaussian HMM for XAUUSD
Author: Strategy Code Engineer Agent
Date: 2026-03-08
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

MODEL_DIR = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Fixed seeds for reproducibility
RANDOM_SEED = 42

# Regime label constants
REGIME_BULL     = "bull"
REGIME_BEAR     = "bear"
REGIME_SIDEWAYS = "sideways"


class XAUUSDRegimeModel:
    """
    GaussianHMM-based regime detector for XAUUSD.

    Detects 3 market regimes (bull / bear / sideways) from:
    - Lagged log returns
    - Lagged normalized RSI
    - Lagged normalized ATR
    - Lagged realized volatility

    All features are pre-shifted by 1 bar in features.py to prevent lookahead.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200):
        self.n_states  = n_states
        self.n_iter    = n_iter
        self.model     = None
        self.scaler    = StandardScaler()
        self.state_map: Dict[str, int] = {}   # {"bull": k, "bear": k, "sideways": k}
        self.fitted    = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "XAUUSDRegimeModel":
        """
        Fit HMM on feature matrix X (n_samples, n_features).
        All rows must be NaN-free (call get_hmm_feature_matrix first).
        """
        np.random.seed(RANDOM_SEED)
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=RANDOM_SEED,
            tol=1e-4,
        )
        self.model.fit(X_scaled)
        self._map_states(X)
        self.fitted = True
        return self

    def _map_states(self, X: np.ndarray) -> None:
        """
        Map HMM integer states to regime labels by mean return.
        - 2 states: bear (lowest return) + bull (highest return)
        - 3 states: bear + sideways + bull
        """
        X_scaled   = self.scaler.transform(X)
        states     = self.model.predict(X_scaled)
        mean_rets  = {}
        for s in range(self.n_states):
            mask = (states == s)
            if mask.sum() > 0:
                mean_rets[s] = X[mask, 0].mean()   # col 0 = log return
            else:
                mean_rets[s] = 0.0

        sorted_states = sorted(mean_rets, key=mean_rets.get)
        if self.n_states == 2:
            self.state_map = {
                REGIME_BEAR: sorted_states[0],
                REGIME_BULL: sorted_states[1],
            }
        else:
            self.state_map = {
                REGIME_BEAR:     sorted_states[0],
                REGIME_SIDEWAYS: sorted_states[1],
                REGIME_BULL:     sorted_states[2],
            }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer state array for feature matrix X."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability matrix (n_samples, n_states)."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def regime_labels(self, X: np.ndarray) -> pd.Series:
        """Return Series of regime labels: 'bull', 'bear', 'sideways'."""
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        labels  = pd.Series([inv_map.get(s, "unknown") for s in states])
        return labels

    def bull_probability(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability of bull regime for each observation."""
        proba      = self.predict_proba(X)
        bull_state = self.state_map[REGIME_BULL]
        return proba[:, bull_state]

    def bear_probability(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability of bear regime for each observation."""
        proba      = self.predict_proba(X)
        bear_state = self.state_map[REGIME_BEAR]
        return proba[:, bear_state]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, name: str = "hmm_regime_model") -> Path:
        """Save model, scaler, and state map to models/ directory."""
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump({
            "model":     self.model,
            "scaler":    self.scaler,
            "state_map": self.state_map,
            "n_states":  self.n_states,
            "n_iter":    self.n_iter,
        }, path)
        print(f"[hmm_model] Saved to {path}")
        return path

    @classmethod
    def load(cls, name: str = "hmm_regime_model") -> "XAUUSDRegimeModel":
        """Load model from models/ directory."""
        path = MODEL_DIR / f"{name}.pkl"
        data = joblib.load(path)
        obj  = cls(n_states=data["n_states"], n_iter=data["n_iter"])
        obj.model     = data["model"]
        obj.scaler    = data["scaler"]
        obj.state_map = data["state_map"]
        obj.fitted    = True
        print(f"[hmm_model] Loaded from {path}")
        return obj

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self, X: np.ndarray) -> None:
        """Print regime statistics."""
        states    = self.predict(X)
        inv_map   = {v: k for k, v in self.state_map.items()}
        print(f"\n[HMM Regime Summary] n_states={self.n_states}")
        print(f"State map: {self.state_map}")
        for s in range(self.n_states):
            mask     = (states == s)
            label    = inv_map.get(s, "unknown")
            pct      = mask.mean() * 100
            mean_ret = X[mask, 0].mean() if mask.sum() > 0 else 0.0
            print(f"  State {s} ({label:>10}): {pct:5.1f}% of bars | mean_log_ret={mean_ret:.6f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.loader import load_split
    from src.data.features import add_features, get_hmm_feature_matrix

    train, val, test = load_split("1H")
    train_feat = add_features(train)
    X_train, idx_train = get_hmm_feature_matrix(train_feat)

    model = XAUUSDRegimeModel(n_states=3)
    model.fit(X_train)
    model.summary(X_train)
    model.save("hmm_regime_model")

    # Test reload
    loaded = XAUUSDRegimeModel.load("hmm_regime_model")
    loaded.summary(X_train)
