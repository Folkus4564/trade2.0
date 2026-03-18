"""
live/signal_pipeline.py - Full signal pipeline for live bars.

Wraps the existing backtest feature/signal chain:
  add_1h_features -> get_hmm_feature_matrix -> HMM predict
  -> add_5m_features -> forward_fill_1h_regime
  -> route_signals -> compute_stops_regime_aware

Returns the last bar's signal state as a structured dict.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from trade2.features.builder import add_1h_features, add_5m_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime
from trade2.signals.router import route_signals, ffill_tv_cols_to_5m
from trade2.signals.generator import compute_stops_regime_aware

logger = logging.getLogger(__name__)


class SignalPipeline:
    """
    Runs the full feature+regime+signal chain on a rolling bar window.

    Usage:
        pipeline = SignalPipeline(hmm_model, config)
        state = pipeline.run(df_1h, df_5m)
        # state["signal_long"] == 1 -> open long
    """

    def __init__(self, hmm_model: XAUUSDRegimeModel, config: Dict[str, Any]):
        self.hmm   = hmm_model
        self.cfg   = config

    def run(
        self,
        df_1h: pd.DataFrame,
        df_5m: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on rolling windows and return the latest bar state.

        Args:
            df_1h: 1H OHLCV DataFrame (DatetimeIndex UTC, at least 100 bars)
            df_5m: 5M OHLCV DataFrame (DatetimeIndex UTC, at least 100 bars)

        Returns:
            Dict with keys:
                bar_time, signal_long, signal_short, exit_long, exit_short,
                stop_long, stop_short, tp_long, tp_short,
                position_size_long, position_size_short,
                regime, bull_prob, bear_prob,
                trailing_atr_mult_long, trailing_atr_mult_short,
                signal_source, atr_1h, close_5m
        """
        # ---- Step 1: 1H features ----
        reg_feat = add_1h_features(df_1h, self.cfg)

        # ---- Step 2: HMM feature matrix ----
        X, hmm_idx = get_hmm_feature_matrix(reg_feat, self.cfg)
        if len(X) == 0:
            logger.warning("[SignalPipeline] No valid HMM features — returning flat state")
            return self._flat_state(df_5m.index[-1] if len(df_5m) > 0 else pd.Timestamp.now(tz="UTC"))

        # ---- Step 3: HMM predict ----
        hmm_labels    = self.hmm.regime_labels(X)
        hmm_bull_prob = self.hmm.bull_probability(X)
        hmm_bear_prob = self.hmm.bear_probability(X)

        # ---- Step 4: 5M features ----
        sig_feat = add_5m_features(df_5m, self.cfg)

        # ---- Step 5: Forward-fill 1H regime onto 5M bars ----
        atr_1h_series      = reg_feat["atr_14"] if "atr_14" in reg_feat.columns else None
        hma_rising_series  = reg_feat["hma_rising"].astype(bool) if "hma_rising" in reg_feat.columns else None
        pah_series         = reg_feat["price_above_hma"].astype(bool) if "price_above_hma" in reg_feat.columns else None

        df_sig = forward_fill_1h_regime(
            sig_feat,
            hmm_labels,
            hmm_bull_prob,
            hmm_bear_prob,
            hmm_idx,
            atr_1h=atr_1h_series,
            hma_rising=hma_rising_series,
            price_above_hma=pah_series,
        )

        # Forward-fill any TV indicator bull/bear columns from 1H to 5M
        df_sig = ffill_tv_cols_to_5m(df_sig, reg_feat)

        # ---- Step 6: Route signals ----
        df_sig = route_signals(df_sig, self.cfg)

        # ---- Step 7: Compute regime-aware SL/TP ----
        df_sig = compute_stops_regime_aware(df_sig, self.cfg)

        # ---- Step 8: Extract last bar ----
        if len(df_sig) == 0:
            return self._flat_state(df_5m.index[-1])

        last = df_sig.iloc[-1]
        return {
            "bar_time":                 df_sig.index[-1],
            "signal_long":              int(last.get("signal_long", 0)),
            "signal_short":             int(last.get("signal_short", 0)),
            "exit_long":                int(last.get("exit_long", 0)),
            "exit_short":               int(last.get("exit_short", 0)),
            "stop_long":                float(last.get("stop_long", 0.0)),
            "stop_short":               float(last.get("stop_short", 0.0)),
            "tp_long":                  float(last.get("tp_long", 0.0)),
            "tp_short":                 float(last.get("tp_short", 0.0)),
            "position_size_long":       float(last.get("position_size_long", 0.5)),
            "position_size_short":      float(last.get("position_size_short", 0.5)),
            "regime":                   str(last.get("regime", "sideways")),
            "bull_prob":                float(last.get("bull_prob", 0.0)),
            "bear_prob":                float(last.get("bear_prob", 0.0)),
            "trailing_atr_mult_long":   float(last.get("trailing_atr_mult_long", 0.0)),
            "trailing_atr_mult_short":  float(last.get("trailing_atr_mult_short", 0.0)),
            "signal_source":            str(last.get("signal_source", "")),
            "atr_1h":                   float(last.get("atr_1h", last.get("atr_14", 0.0))),
            "close_5m":                 float(last.get("Close", 0.0)),
        }

    def reload_model(self, hmm_model: XAUUSDRegimeModel) -> None:
        """Hot-swap the HMM model (used after weekly retrain)."""
        self.hmm = hmm_model
        logger.info("[SignalPipeline] HMM model reloaded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_state(bar_time: pd.Timestamp) -> Dict[str, Any]:
        return {
            "bar_time":                 bar_time,
            "signal_long":              0,
            "signal_short":             0,
            "exit_long":                1,
            "exit_short":               1,
            "stop_long":                0.0,
            "stop_short":               0.0,
            "tp_long":                  0.0,
            "tp_short":                 0.0,
            "position_size_long":       0.5,
            "position_size_short":      0.5,
            "regime":                   "sideways",
            "bull_prob":                0.0,
            "bear_prob":                0.0,
            "trailing_atr_mult_long":   0.0,
            "trailing_atr_mult_short":  0.0,
            "signal_source":            "",
            "atr_1h":                   0.0,
            "close_5m":                 0.0,
        }
