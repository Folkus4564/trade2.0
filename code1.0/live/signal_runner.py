"""
Module: signal_runner.py
Purpose: SignalRunner — load approved model+config, process live candles, emit signals
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from typing import Any, Dict, Optional


class SignalRunner:
    """
    Loads a trained model from approved_strategies/ and processes incoming candles
    to emit trading signals compatible with OrderManager.

    Usage:
        runner = SignalRunner("approved_strategies/xauusd_smc_hmm_regime_2026_03_10")
        runner.load()
        signal = runner.process_bar(latest_bars_df)
    """

    def __init__(self, strategy_dir: str):
        """
        Args:
            strategy_dir: Path to an approved_strategies/<name>/ directory
        """
        self.strategy_dir = Path(strategy_dir)
        self.model        = None
        self.config       = None
        self._loaded      = False

    def load(self) -> None:
        """
        Load model.pkl and config.yaml from the approved strategy directory.
        """
        raise NotImplementedError("load() not implemented")

    def process_bar(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the latest bars DataFrame and return a signal dict.

        Args:
            bars: Recent OHLCV bars (enough history for all features)

        Returns:
            Signal dict: {
                "direction": "long" | "short" | "flat",
                "size":      float,   # 0.0 to 1.0 fraction of account
                "stop":      float | None,
                "tp":        float | None,
                "regime":    str,
                "confidence": float,
            }
        """
        if not self._loaded:
            raise RuntimeError("Call load() before process_bar()")
        raise NotImplementedError("process_bar() not implemented")

    def warmup_bars_needed(self) -> int:
        """
        Return the number of historical bars needed to compute all features.
        At minimum: max(HMA period, BB period, ATR period, DC period) + HMM warmup.
        """
        raise NotImplementedError("warmup_bars_needed() not implemented")
