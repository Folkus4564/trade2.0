"""
scripts/download_data.py - Download XAUUSD data from Dukascopy.
Consolidates src_v2/data/dukascopy_downloader.py functionality.

Usage:
    python scripts/download_data.py --timeframes 1H 5M --start 2019-01-01 --end 2025-06-30
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports before package install
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def download(timeframes: list, start: str, end: str, output_dir: Path) -> None:
    """Download OHLCV data from Dukascopy for given timeframes and date range."""
    try:
        from src_v2.data.dukascopy_downloader import download_ohlcv
    except ImportError:
        print("[download] src_v2 downloader not found. Place dukascopy_downloader.py in src_v2/data/")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for tf in timeframes:
        print(f"\n[download] Downloading {tf} data from {start} to {end}...")
        try:
            out_path = output_dir / f"XAUUSD_{tf}_{start[:4]}_to_{end[:4]}.csv"
            download_ohlcv(
                symbol    = "XAUUSD",
                timeframe = tf,
                start     = start,
                end       = end,
                output    = str(out_path),
            )
            print(f"[download] Saved to {out_path}")
        except Exception as e:
            print(f"[download] Failed for {tf}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download XAUUSD data from Dukascopy")
    parser.add_argument("--timeframes", nargs="+", default=["1H", "5M"], help="Timeframes to download")
    parser.add_argument("--start",      default="2019-01-01",            help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",        default="2025-06-30",            help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/raw",              help="Output directory")
    args = parser.parse_args()

    download(
        timeframes  = args.timeframes,
        start       = args.start,
        end         = args.end,
        output_dir  = PROJECT_ROOT / args.output_dir,
    )
