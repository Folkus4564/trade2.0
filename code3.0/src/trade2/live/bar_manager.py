"""
live/bar_manager.py - New-bar detection and rolling window management.

Detects when a new 5M bar has closed by comparing the latest bar timestamp
to the last-seen timestamp. Maintains separate rolling windows for 1H and 5M.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd

from trade2.live.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class BarManager:
    """
    Tracks new 5M bar closes and provides fresh OHLCV windows for the pipeline.

    Usage:
        bm = BarManager(connector, warmup_1h=200, warmup_5m=300)
        bm.initialize()

        while True:
            if bm.poll_new_bar():
                df_1h, df_5m = bm.get_windows()
                # run signal pipeline ...
            time.sleep(10)
    """

    def __init__(
        self,
        connector: MT5Connector,
        warmup_1h: int = 200,
        warmup_5m: int = 300,
        stale_warn_sec: int = 120,
        stale_pause_sec: int = 600,
    ):
        self.connector      = connector
        self.warmup_1h      = warmup_1h
        self.warmup_5m      = warmup_5m
        self.stale_warn_sec = stale_warn_sec
        self.stale_pause_sec = stale_pause_sec

        self._last_5m_ts: Optional[pd.Timestamp] = None
        self._df_1h: Optional[pd.DataFrame] = None
        self._df_5m: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Fetch initial warmup windows and record last-seen 5M timestamp."""
        logger.info("[BarManager] Initializing warmup windows ...")
        self._df_1h = self.connector.fetch_bars("1H", self.warmup_1h)
        self._df_5m = self.connector.fetch_bars("5M", self.warmup_5m)
        self._last_5m_ts = self._df_5m.index[-1]
        logger.info(
            f"[BarManager] Ready | 1H bars={len(self._df_1h)} | "
            f"5M bars={len(self._df_5m)} | last_5m={self._last_5m_ts}"
        )

    def poll_new_bar(self) -> bool:
        """
        Fetch latest bars and detect if a new 5M bar has closed.

        Returns True if a new bar was detected (windows updated).
        """
        df_1h_new = self.connector.fetch_bars("1H", self.warmup_1h)
        df_5m_new = self.connector.fetch_bars("5M", self.warmup_5m)

        latest_ts = df_5m_new.index[-1]

        # Staleness check
        age_sec = self._bar_age_seconds(latest_ts)
        if age_sec > self.stale_pause_sec:
            logger.warning(f"[BarManager] Data stale {age_sec:.0f}s — pausing signal processing")
            return False
        if age_sec > self.stale_warn_sec:
            logger.warning(f"[BarManager] Data stale {age_sec:.0f}s — check MT5 connection")

        if self._last_5m_ts is None or latest_ts > self._last_5m_ts:
            self._df_1h = df_1h_new
            self._df_5m = df_5m_new
            self._last_5m_ts = latest_ts
            logger.debug(f"[BarManager] New 5M bar: {latest_ts}")
            return True

        return False

    def get_windows(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df_1h, df_5m) current rolling windows."""
        if self._df_1h is None or self._df_5m is None:
            raise RuntimeError("BarManager not initialized. Call initialize() first.")
        return self._df_1h.copy(), self._df_5m.copy()

    @property
    def last_5m_ts(self) -> Optional[pd.Timestamp]:
        return self._last_5m_ts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_age_seconds(bar_ts: pd.Timestamp) -> float:
        now_utc = pd.Timestamp.now(tz="UTC")
        bar_utc = bar_ts if bar_ts.tzinfo is not None else bar_ts.tz_localize("UTC")
        return (now_utc - bar_utc).total_seconds()
