"""
Module: candle_fetcher.py
Purpose: Abstract CandleFetcher — skeleton for fetching live OHLCV bars
"""

import pandas as pd
from typing import Optional


class CandleFetcher:
    """
    Abstract interface for fetching live OHLCV candles from a broker or data feed.
    """

    def connect(self, config: dict) -> None:
        """Connect to data feed."""
        raise NotImplementedError("connect() not implemented")

    def fetch_latest(
        self,
        instrument: str,
        timeframe:  str,       # e.g. "1H"
        n_bars:     int = 500, # number of historical bars to fetch
    ) -> pd.DataFrame:
        """
        Fetch the latest N OHLCV bars.

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume]
            and DatetimeIndex (UTC).
        """
        raise NotImplementedError("fetch_latest() not implemented")

    def subscribe(self, instrument: str, callback) -> None:
        """
        Subscribe to real-time bar updates.
        callback(bar: pd.Series) is called on each completed bar.
        """
        raise NotImplementedError("subscribe() not implemented")

    def unsubscribe(self, instrument: str) -> None:
        """Unsubscribe from real-time bar updates."""
        raise NotImplementedError("unsubscribe() not implemented")
