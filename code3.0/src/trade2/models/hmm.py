"""
models/hmm.py - Unified HMM regime detector.
Uses v1's k-means initialization (stable convergence) + v2's state_distribution().
No module-level mkdir. Model path passed explicitly.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Optional

REGIME_BULL     = "bull"
REGIME_BEAR     = "bear"
REGIME_SIDEWAYS = "sideways"


class XAUUSDRegimeModel:
    """
    GaussianHMM-based regime detector for XAUUSD.

    Detects 3 market regimes (bull / bear / sideways) using 7 lag-safe features.
    Uses k-means initialization for stable, reproducible convergence.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_seed: int = 42):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.random_seed = random_seed
        self.model       = None
        self.scaler      = StandardScaler()
        self.state_map:  Dict[str, int] = {}
        self.fitted      = False

    def fit(self, X: np.ndarray) -> "XAUUSDRegimeModel":
        """
        Fit HMM with k-means initialization for stable convergence.

        Args:
            X: Feature matrix (NaN-free rows, shape [n_samples, n_features])
        """
        X_scaled = self.scaler.fit_transform(X)

        # K-means initialization
        km = KMeans(n_clusters=self.n_states, random_state=self.random_seed, n_init=10)
        km.fit(X_scaled)
        init_means = km.cluster_centers_

        self.model = GaussianHMM(
            n_components  = self.n_states,
            covariance_type = "full",
            n_iter        = self.n_iter,
            random_state  = self.random_seed,
            tol           = 1e-4,
            init_params   = "stc",
            params        = "stmc",
        )
        self.model.means_init = init_means
        self.model.fit(X_scaled)

        self._map_states(X)
        self.fitted = True

        # Warn if any state occupies < 5%
        states = self.model.predict(X_scaled)
        for s in range(self.n_states):
            pct = (states == s).mean()
            if pct < 0.05:
                print(f"[hmm] WARNING: state {s} only {pct*100:.1f}% of bars")

        return self

    def _map_states(self, X: np.ndarray) -> None:
        """Map integer states to labels by mean log return (column 0)."""
        X_scaled = self.scaler.transform(X)
        states   = self.model.predict(X_scaled)
        mean_rets = {}
        for s in range(self.n_states):
            mask = (states == s)
            mean_rets[s] = X[mask, 0].mean() if mask.sum() > 0 else 0.0

        sorted_states = sorted(mean_rets, key=mean_rets.get)
        if self.n_states == 2:
            self.state_map = {REGIME_BEAR: sorted_states[0], REGIME_BULL: sorted_states[1]}
        else:
            self.state_map = {
                REGIME_BEAR:     sorted_states[0],
                REGIME_SIDEWAYS: sorted_states[1],
                REGIME_BULL:     sorted_states[2],
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer state array."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability matrix (n_samples, n_states)."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(self.scaler.transform(X))

    def regime_labels(self, X: np.ndarray) -> pd.Series:
        """Return Series of regime labels: 'bull', 'bear', 'sideways'."""
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        return pd.Series([inv_map.get(s, "unknown") for s in states])

    def bull_probability(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability of bull regime."""
        return self.predict_proba(X)[:, self.state_map[REGIME_BULL]]

    def bear_probability(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability of bear regime."""
        return self.predict_proba(X)[:, self.state_map[REGIME_BEAR]]

    def sideways_probability(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability of sideways regime. Returns zeros for 2-state HMM."""
        if REGIME_SIDEWAYS not in self.state_map:
            return np.zeros(len(X))
        return self.predict_proba(X)[:, self.state_map[REGIME_SIDEWAYS]]

    def self_transition_prob(self, regime_label: str) -> float:
        """
        Return self-transition probability for a regime (from the HMM transition matrix).
        This is model.transmat_[state_idx, state_idx] -- probability of staying in the regime.
        A low value means the HMM learned that this regime is historically short-lived.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if regime_label not in self.state_map:
            raise KeyError(f"Unknown regime '{regime_label}'. Known: {list(self.state_map)}")
        idx = self.state_map[regime_label]
        return float(self.model.transmat_[idx, idx])

    def warm_update(self, X_new: np.ndarray, n_iter: int = 30) -> "XAUUSDRegimeModel":
        """
        Warm-start update: initialise a new HMM from existing parameters,
        then run EM on X_new only.  The existing scaler is reused so the
        feature space stays consistent with the original training.

        Args:
            X_new: Recent feature matrix (NaN-free, same feature order as fit())
            n_iter: Number of EM iterations (fewer than full retrain)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X_new)

        new_hmm = GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "full",
            n_iter          = n_iter,
            random_state    = self.random_seed,
            tol             = 1e-4,
            init_params     = "",    # skip random init — we copy params below
            params          = "stmc",
        )
        # Seed from current model
        new_hmm.startprob_ = self.model.startprob_.copy()
        new_hmm.transmat_  = self.model.transmat_.copy()
        new_hmm.means_     = self.model.means_.copy()
        new_hmm.covars_    = self.model.covars_.copy()

        new_hmm.fit(X_scaled)
        self.model  = new_hmm
        self._map_states(X_new)

        states = self.model.predict(X_scaled)
        for s in range(self.n_states):
            pct = (states == s).mean()
            if pct < 0.05:
                print(f"[hmm] WARNING: state {s} only {pct*100:.1f}% of bars after warm update")

        return self

    def score(self, X: np.ndarray) -> float:
        """Return log-likelihood per sample (higher is better)."""
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.score(self.scaler.transform(X))

    def state_distribution(self, X: np.ndarray) -> Dict[str, float]:
        """Return fraction of bars in each regime state."""
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        return {inv_map.get(s, "unknown"): float((states == s).mean()) for s in range(self.n_states)}

    def save(self, path: Path) -> Path:
        """Save model, scaler, and state map to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":       self.model,
            "scaler":      self.scaler,
            "state_map":   self.state_map,
            "n_states":    self.n_states,
            "n_iter":      self.n_iter,
            "random_seed": self.random_seed,
        }, path)
        print(f"[hmm] Saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "XAUUSDRegimeModel":
        """Load model from explicit path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        obj  = cls(
            n_states    = data["n_states"],
            n_iter      = data["n_iter"],
            random_seed = data.get("random_seed", 42),
        )
        obj.model     = data["model"]
        obj.scaler    = data["scaler"]
        obj.state_map = data["state_map"]
        obj.fitted    = True
        print(f"[hmm] Loaded from {path}")
        return obj

    def summary(self, X: np.ndarray) -> None:
        """Print regime distribution statistics."""
        states  = self.predict(X)
        inv_map = {v: k for k, v in self.state_map.items()}
        print(f"\n[HMM Summary] n_states={self.n_states}")
        print(f"State map: {self.state_map}")
        for s in range(self.n_states):
            mask     = (states == s)
            label    = inv_map.get(s, "unknown")
            pct      = mask.mean() * 100
            mean_ret = X[mask, 0].mean() if mask.sum() > 0 else 0.0
            print(f"  State {s} ({label:>10}): {pct:5.1f}% | mean_log_ret={mean_ret:.6f}")
