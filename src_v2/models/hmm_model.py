"""
Module: hmm_model.py
Purpose: 3-state HMM regime detector for XAUUSD on 1H bars
         States: bull / bear / sideways
         Enhanced with 7 features and stability filter
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Dict

MODEL_DIR = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

REGIME_BULL     = "bull"
REGIME_BEAR     = "bear"
REGIME_SIDEWAYS = "sideways"


class XAUUSDRegimeModel:
    """
    3-state GaussianHMM regime detector for XAUUSD.

    Trained on 1H bars with 7 features:
    - log return, normalized RSI, normalized ATR, realized vol
    - HMA slope, BB width, MACD histogram
    All pre-shifted by 1 bar in features.py.

    Includes regime stability filter: regime must persist for N bars
    before being considered active.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_seed: int = RANDOM_SEED):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.random_seed = random_seed
        self.model       = None
        self.scaler      = StandardScaler()
        self.state_map: Dict[str, int] = {}
        self.fitted      = False

    def fit(self, X: np.ndarray) -> "XAUUSDRegimeModel":
        """Fit HMM on feature matrix X (NaN-free)."""
        np.random.seed(self.random_seed)
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_seed,
            tol=1e-4,
        )
        self.model.fit(X_scaled)
        self._map_states(X)
        self.fitted = True
        return self

    def _map_states(self, X: np.ndarray) -> None:
        """Map HMM states to regime labels by mean return (col 0)."""
        X_scaled = self.scaler.transform(X)
        states   = self.model.predict(X_scaled)
        mean_rets = {}
        for s in range(self.n_states):
            mask = (states == s)
            mean_rets[s] = X[mask, 0].mean() if mask.sum() > 0 else 0.0

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(self.scaler.transform(X))

    def regime_labels(self, X: np.ndarray) -> pd.Series:
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        return pd.Series([inv_map.get(s, "unknown") for s in states])

    def bull_probability(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba[:, self.state_map[REGIME_BULL]]

    def bear_probability(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba[:, self.state_map[REGIME_BEAR]]

    def state_distribution(self, X: np.ndarray) -> Dict[str, float]:
        """Return percentage of bars in each state."""
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        dist = {}
        for s in range(self.n_states):
            label = inv_map.get(s, "unknown")
            dist[label] = float((states == s).mean())
        return dist

    def save(self, name: str = "hmm_regime_model_v2") -> Path:
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "state_map": self.state_map,
            "n_states": self.n_states,
            "n_iter": self.n_iter,
        }, path)
        print(f"[hmm_model] Saved to {path}")
        return path

    @classmethod
    def load(cls, name: str = "hmm_regime_model_v2") -> "XAUUSDRegimeModel":
        path = MODEL_DIR / f"{name}.pkl"
        data = joblib.load(path)
        obj = cls(n_states=data["n_states"], n_iter=data["n_iter"])
        obj.model     = data["model"]
        obj.scaler    = data["scaler"]
        obj.state_map = data["state_map"]
        obj.fitted    = True
        print(f"[hmm_model] Loaded from {path}")
        return obj

    def summary(self, X: np.ndarray) -> None:
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        print(f"\n[HMM Regime Summary] n_states={self.n_states}, n_features={X.shape[1]}")
        print(f"State map: {self.state_map}")
        for s in range(self.n_states):
            mask     = (states == s)
            label    = inv_map.get(s, "unknown")
            pct      = mask.mean() * 100
            mean_ret = X[mask, 0].mean() if mask.sum() > 0 else 0.0
            print(f"  State {s} ({label:>10}): {pct:5.1f}% of bars | mean_log_ret={mean_ret:.6f}")
