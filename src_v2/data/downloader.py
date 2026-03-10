"""
Module: downloader.py
Purpose: Download XAUUSD data from Dukascopy at any timeframe
"""

from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_xauusd(
    timeframe: str = "M5",
    start: date = date(2019, 1, 1),
    end: date = date(2024, 12, 31),
    threads: int = 4,
) -> Path:
    """
    Download XAUUSD OHLCV data from Dukascopy.

    Args:
        timeframe: Duka timeframe string (M1, M5, M15, M30, H1, H4, D1)
        start:     Start date
        end:       End date
        threads:   Number of download threads

    Returns:
        Path to the downloaded CSV
    """
    from duka.app import app
    from duka.core.utils import TimeFrame

    tf_map = {
        "M1": TimeFrame.M1,
        "M5": TimeFrame.M5,
        "M15": TimeFrame.M15,
        "M30": TimeFrame.M30,
        "H1": TimeFrame.H1,
        "H4": TimeFrame.H4,
        "D1": TimeFrame.D1,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use one of: {list(tf_map.keys())}")

    tf_obj = tf_map[timeframe]

    print(f"[downloader] Downloading XAUUSD {timeframe} from {start} to {end}...")
    app(
        symbols=["XAUUSD"],
        start=start,
        end=end,
        threads=threads,
        timeframe=tf_obj,
        folder=str(RAW_DIR),
        header=True,
    )

    # Duka saves as XAUUSD-{timeframe}-{start}-{end}.csv or similar
    # Find the most recent CSV matching the pattern
    candidates = sorted(RAW_DIR.glob("XAUUSD*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("Download completed but no CSV found in raw/")

    result = candidates[0]
    print(f"[downloader] Saved: {result} ({result.stat().st_size / 1e6:.1f} MB)")
    return result


if __name__ == "__main__":
    # Download 5min data for v2 strategy
    download_xauusd("M5", date(2019, 1, 1), date(2024, 12, 31))
