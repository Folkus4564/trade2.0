"""
live/data_accumulator.py - Append new live bars to local CSV files.

On each new 1H/5M bar close, appends it to the live accumulation CSV.
Deduplicates by timestamp (idempotent — safe if bars arrive twice).
At retrain time, the full dataset = original + live CSV.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


class DataAccumulator:
    """
    Accumulates live MT5 bars to growing CSV files.

    Two files maintained:
    - XAUUSD_1H_live.csv  — new 1H bars
    - XAUUSD_5M_live.csv  — new 5M bars

    Both are deduplicated on the 'time' index column.
    """

    def __init__(self, live_data_dir: Path):
        self.dir = Path(live_data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path_1h = self.dir / "XAUUSD_1H_live.csv"
        self.path_5m = self.dir / "XAUUSD_5M_live.csv"

        # In-memory caches (to avoid full re-read on every bar)
        self._last_1h_ts: Optional[pd.Timestamp] = None
        self._last_5m_ts: Optional[pd.Timestamp] = None
        self._init_last_timestamps()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_1h(self, df: pd.DataFrame) -> int:
        """
        Append new 1H bars that are newer than the last saved timestamp.

        Args:
            df: Full 1H rolling window from BarManager (sorted ascending)

        Returns:
            Number of new rows appended.
        """
        return self._append(df, self.path_1h, "_last_1h_ts", "1H")

    def append_5m(self, df: pd.DataFrame) -> int:
        """Append new 5M bars newer than the last saved timestamp."""
        return self._append(df, self.path_5m, "_last_5m_ts", "5M")

    def load_full_1h(self, original_csv: Path) -> pd.DataFrame:
        """
        Load original 1H CSV + live accumulation, deduplicated and sorted.

        Used by Retrainer to build the full training dataset.
        """
        return self._load_full(original_csv, self.path_1h, "1H")

    def load_full_5m(self, original_csv: Path) -> pd.DataFrame:
        """Load original 5M CSV + live accumulation."""
        return self._load_full(original_csv, self.path_5m, "5M")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(
        self,
        df: pd.DataFrame,
        path: Path,
        last_ts_attr: str,
        label: str,
    ) -> int:
        last_ts = getattr(self, last_ts_attr)

        # Filter to only new bars
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx, utc=True)
                df = df.copy()
                df.index = idx
            except Exception as e:
                logger.warning(f"[DataAccumulator] Cannot parse {label} index: {e}")
                return 0

        if last_ts is not None:
            new_bars = df[df.index > last_ts]
        else:
            new_bars = df

        if new_bars.empty:
            return 0

        # Ensure columns exist
        cols = [c for c in _REQUIRED_COLS if c in new_bars.columns]
        write_df = new_bars[cols].copy()
        write_df.index.name = "time"

        header = not path.exists() or path.stat().st_size == 0
        write_df.to_csv(path, mode="a", header=header)

        n = len(write_df)
        setattr(self, last_ts_attr, write_df.index[-1])
        logger.debug(f"[DataAccumulator] Appended {n} new {label} bars to {path.name}")
        return n

    def _load_full(
        self,
        original_csv: Path,
        live_csv: Path,
        label: str,
    ) -> pd.DataFrame:
        frames = []

        # Original historical data
        if Path(original_csv).exists():
            try:
                df_orig = pd.read_csv(original_csv, index_col=0, parse_dates=True)
                df_orig.index = pd.to_datetime(df_orig.index, utc=True)
                frames.append(df_orig)
                logger.info(f"[DataAccumulator] Loaded original {label}: {len(df_orig)} bars")
            except Exception as e:
                logger.error(f"[DataAccumulator] Failed to load {original_csv}: {e}")

        # Live accumulated bars
        if live_csv.exists() and live_csv.stat().st_size > 0:
            try:
                df_live = pd.read_csv(live_csv, index_col=0, parse_dates=True)
                df_live.index = pd.to_datetime(df_live.index, utc=True)
                frames.append(df_live)
                logger.info(f"[DataAccumulator] Loaded live {label}: {len(df_live)} bars")
            except Exception as e:
                logger.error(f"[DataAccumulator] Failed to load live {label} CSV: {e}")

        if not frames:
            raise RuntimeError(f"[DataAccumulator] No {label} data available for retrain")

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        logger.info(f"[DataAccumulator] Full {label} dataset: {len(combined)} bars")
        return combined

    def _init_last_timestamps(self) -> None:
        """Read the last timestamp from existing live CSVs (if any)."""
        for path, attr in [(self.path_1h, "_last_1h_ts"), (self.path_5m, "_last_5m_ts")]:
            if path.exists() and path.stat().st_size > 0:
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True, nrows=None)
                    if not df.empty:
                        setattr(self, attr, pd.to_datetime(df.index[-1], utc=True))
                except Exception:
                    pass
