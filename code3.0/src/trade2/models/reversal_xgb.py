"""
models/reversal_xgb.py - XGBoost supervised reversal probability detector.

Purpose:
  At each bar where price is at an SD-extreme zone, predict the probability that
  price will actually reverse (move >= win_multiplier * ATR within forward_bars).

  This is a SUPERVISED alternative to using HMM probability for sizing/gating.
  Labels use forward price data -> ONLY generated on training split, never at
  inference time (no lookahead bias).

Usage:
  model = ReversalXGBModel(config)
  model.fit(train_df, direction="long")   # trains on train split
  probs = model.predict_proba(df)         # inference - uses only lagged features
  model.save(path)
  model = ReversalXGBModel.load(path)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


FEATURE_COLS_LONG = [
    # SD position features
    "sd_smoothed", "sd_zone_float", "sd_momentum",
    "consecutive_extreme",
    # HMM regime
    "bear_prob", "bull_prob",
    # Momentum / strength
    "rsi_14", "adx_14",
    "atr_expansion",
    "ret_1", "ret_5",
    # Candle structure
    "candle_body_ratio",
    "lower_wick_atr",
    "upper_wick_atr",
    # Context
    "in_demand_zone",
    "ob_bullish",
    "hour_sin", "hour_cos",   # time of day (encoded cyclically)
]

FEATURE_COLS_SHORT = [
    # SD position features
    "sd_smoothed", "sd_zone_float", "sd_momentum",
    "consecutive_extreme_short",
    # HMM regime
    "bear_prob", "bull_prob",
    # Momentum / strength
    "rsi_14", "adx_14",
    "atr_expansion",
    "ret_1", "ret_5",
    # Candle structure
    "candle_body_ratio",
    "upper_wick_atr",
    "lower_wick_atr",
    # Context
    "in_supply_zone",
    "ob_bearish",
    "hour_sin", "hour_cos",
]


class ReversalXGBModel:
    """
    XGBoost-based reversal probability detector for SD-mean entries.

    Trains separately for long and short directions.
    Uses isotonic calibration for reliable probabilities.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        xgb_cfg = self._xgb_cfg()

        self.forward_bars   = int(xgb_cfg.get("forward_bars",   20))
        self.win_multiplier = float(xgb_cfg.get("win_multiplier", 0.8))
        self.min_entry_zone = int(xgb_cfg.get("min_entry_zone",  2))
        self.n_estimators   = int(xgb_cfg.get("n_estimators",    200))
        self.max_depth      = int(xgb_cfg.get("max_depth",       4))
        self.learning_rate  = float(xgb_cfg.get("learning_rate", 0.05))
        self.subsample      = float(xgb_cfg.get("subsample",     0.8))
        self.colsample      = float(xgb_cfg.get("colsample_bytree", 0.8))

        self._model_long:  Optional[CalibratedClassifierCV] = None
        self._model_short: Optional[CalibratedClassifierCV] = None
        self._scaler_long  = StandardScaler()
        self._scaler_short = StandardScaler()
        self.fitted_long  = False
        self.fitted_short = False
        self.feature_importance_long:  Optional[Dict] = None
        self.feature_importance_short: Optional[Dict] = None

    def _xgb_cfg(self) -> Dict:
        return (self.config
                .get("strategies", {})
                .get("smc_sd_mean", {})
                .get("reversal_xgb", {}))

    # ------------------------------------------------------------------
    # Label generation (TRAIN ONLY — uses forward price data)
    # ------------------------------------------------------------------

    def make_labels(
        self, df: pd.DataFrame, direction: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trade-outcome labels for a given direction.

        Labels are ONLY valid on training data. Never call this at inference.

        Label = 1 if the trade WINS: TP1 is hit BEFORE SL within forward_bars.
        TP1 = close + tp_mult * ATR  (default 1.0x ATR, matching partial TP level 1)
        SL  = close - sl_mult * ATR  (default 1.5x ATR, matching risk config)

        This is more discriminative than a simple price-bounce label because it
        accounts for the actual risk/reward structure of the strategy:
         - Base win rate ~40-50% (vs 74% for simple bounce label)
         - Model must learn what separates TP hits from SL hits
         - Directly optimizes for what matters: profitable entries

        Args:
            df:        Feature DataFrame with OHLC, atr_14, sd_zone columns.
                       Features should already have shift(1) applied.
            direction: 'long' or 'short'

        Returns:
            (labels, mask) where mask marks bars at SD extremes.
            labels[i] = 1 if trade would win (TP1 hit before SL).
        """
        close  = df["Close"].values.astype(float)
        high   = df["High"].values.astype(float)
        low    = df["Low"].values.astype(float)
        atr    = df["atr_14"].values.astype(float) if "atr_14" in df.columns else np.ones(len(close))

        # sd_zone from feature columns (already shift(1) applied in feature builder)
        if "sd_zone" in df.columns:
            sd_zone = df["sd_zone"].fillna(0).round().astype(int).values
        else:
            sd_zone = np.zeros(len(close), dtype=int)

        xgb_cfg = self._xgb_cfg()
        tp_mult = float(xgb_cfg.get("label_tp_mult", 1.0))   # TP1 level
        sl_mult = float(xgb_cfg.get("label_sl_mult", 1.5))   # SL level

        n = len(close)
        labels = np.zeros(n, dtype=float)
        mask   = np.zeros(n, dtype=bool)

        fb = self.forward_bars

        if direction == "long":
            extreme = sd_zone <= -self.min_entry_zone
            for i in range(n - fb):
                if not extreme[i]:
                    continue
                mask[i] = True
                atr_i  = atr[i] if atr[i] > 0 else 1.0
                tp_px  = close[i] + tp_mult * atr_i
                sl_px  = close[i] - sl_mult * atr_i
                # Walk forward bar by bar: TP or SL hit?
                won = False
                for j in range(i + 1, min(i + fb + 1, n)):
                    if high[j] >= tp_px:
                        won = True
                        break
                    if low[j] <= sl_px:
                        break   # SL hit first → loss
                labels[i] = 1.0 if won else 0.0
        else:  # short
            extreme = sd_zone >= self.min_entry_zone
            for i in range(n - fb):
                if not extreme[i]:
                    continue
                mask[i] = True
                atr_i  = atr[i] if atr[i] > 0 else 1.0
                tp_px  = close[i] - tp_mult * atr_i
                sl_px  = close[i] + sl_mult * atr_i
                won = False
                for j in range(i + 1, min(i + fb + 1, n)):
                    if low[j] <= tp_px:
                        won = True
                        break
                    if high[j] >= sl_px:
                        break
                labels[i] = 1.0 if won else 0.0

        return labels, mask

    # ------------------------------------------------------------------
    # Feature matrix builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_X(df: pd.DataFrame, direction: str) -> pd.DataFrame:
        """Extract feature matrix for the given direction."""
        cols = FEATURE_COLS_LONG if direction == "long" else FEATURE_COLS_SHORT
        avail = [c for c in cols if c in df.columns]
        X = df[avail].copy()
        # Fill missing columns with 0
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols].fillna(0.0)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ReversalXGBModel":
        """
        Train both long and short reversal detectors on the training DataFrame.

        Args:
            df: Full training 5M feature DataFrame (OHLC + all features pre-computed).
        """
        for direction in ("long", "short"):
            self._fit_direction(df, direction)
        return self

    def _fit_direction(self, df: pd.DataFrame, direction: str) -> None:
        labels, mask = self.make_labels(df, direction)
        if mask.sum() < 50:
            print(f"  [reversal_xgb] WARNING: only {mask.sum()} {direction} samples, skipping fit")
            return

        X_full = self._build_X(df, direction)
        X = X_full.values[mask]
        y = labels[mask]

        pos_rate = y.mean()
        print(f"  [reversal_xgb] {direction}: {mask.sum()} samples | "
              f"pos_rate={pos_rate:.3f} | features={X_full.shape[1]}")

        # Class weight to handle imbalance
        scale_pos = (1 - pos_rate) / (pos_rate + 1e-9)

        base = XGBClassifier(
            n_estimators    = self.n_estimators,
            max_depth       = self.max_depth,
            learning_rate   = self.learning_rate,
            subsample       = self.subsample,
            colsample_bytree= self.colsample,
            scale_pos_weight= scale_pos,
            eval_metric     = "logloss",
            random_state    = 42,
            n_jobs          = -1,
            verbosity       = 0,
        )

        # Isotonic calibration for reliable probabilities
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)

        if direction == "long":
            scaler = self._scaler_long
        else:
            scaler = self._scaler_short

        X_scaled = scaler.fit_transform(X)
        calibrated.fit(X_scaled, y)

        # Feature importance from base estimator
        try:
            fi = calibrated.estimators_[0].feature_importances_
            cols = FEATURE_COLS_LONG if direction == "long" else FEATURE_COLS_SHORT
            avail = [c for c in cols if c in self._build_X(df, direction).columns]
            fi_dict = dict(zip(avail, fi))
            fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: -x[1]))
            print(f"  [reversal_xgb] {direction} top features: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in list(fi_sorted.items())[:5]))
            if direction == "long":
                self.feature_importance_long = fi_sorted
            else:
                self.feature_importance_short = fi_sorted
        except Exception:
            pass

        if direction == "long":
            self._model_long  = calibrated
            self.fitted_long  = True
        else:
            self._model_short = calibrated
            self.fitted_short = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba_long(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict reversal probability for long entries.
        Returns array of shape (N,) with probabilities in [0, 1].
        Uses ONLY features available at signal time (no future data).
        """
        if not self.fitted_long:
            return np.full(len(df), 0.5)
        X = self._build_X(df, "long").values
        X_scaled = self._scaler_long.transform(X)
        proba = self._model_long.predict_proba(X_scaled)
        # Column 1 = probability of class 1 (reversal happens)
        return proba[:, 1].astype(float)

    def predict_proba_short(self, df: pd.DataFrame) -> np.ndarray:
        """Predict reversal probability for short entries."""
        if not self.fitted_short:
            return np.full(len(df), 0.5)
        X = self._build_X(df, "short").values
        X_scaled = self._scaler_short.transform(X)
        proba = self._model_short.predict_proba(X_scaled)
        return proba[:, 1].astype(float)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save entire model (both directions) to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  [reversal_xgb] Saved model -> {path}")

    @staticmethod
    def load(path: str) -> "ReversalXGBModel":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"  [reversal_xgb] Loaded model <- {path}")
        return model
