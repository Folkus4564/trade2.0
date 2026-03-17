"""
Download XAUUSD data for Jul 2025 - Mar 2026 and append to existing CSVs.
"""
import sys
sys.path.insert(0, "C:/Users/LENOVO/Desktop/trade2.0/code2.0")

from src_v2.data.dukascopy_downloader import download_xauusd_5m
from datetime import date
from pathlib import Path
import pandas as pd

RAW_DIR = Path("C:/Users/LENOVO/Desktop/trade2.0/data/raw")

# Download 5M bars for 2025-07-01 to 2026-03-31
print("[main] Downloading 5M bars: 2025-07-01 to 2026-03-31")
new_5m_path = RAW_DIR / "XAUUSD_5M_2025-07_2026-03.csv"
download_xauusd_5m(
    start=date(2025, 7, 1),
    end=date(2026, 3, 31),
    output_path=new_5m_path,
    workers=100,
    freq="5min",
)

# Also download 1H
print("[main] Downloading 1H bars: 2025-07-01 to 2026-03-31")
new_1h_path = RAW_DIR / "XAUUSD_1H_2025-07_2026-03.csv"
download_xauusd_5m(
    start=date(2025, 7, 1),
    end=date(2026, 3, 31),
    output_path=new_1h_path,
    workers=100,
    freq="1h",
)

# Append to existing 5M
existing_5m = pd.read_csv(RAW_DIR / "XAUUSD_5M_2019_2025.csv")
new_5m = pd.read_csv(new_5m_path)
ts_col_5m = "UTC" if "UTC" in existing_5m.columns else existing_5m.columns[0]
ts_col_new = "UTC" if "UTC" in new_5m.columns else new_5m.columns[0]
new_5m = new_5m.rename(columns={ts_col_new: ts_col_5m})
combined_5m = pd.concat([existing_5m, new_5m]).drop_duplicates(subset=[ts_col_5m]).sort_values(ts_col_5m)
out_5m = RAW_DIR / "XAUUSD_5M_2019_2026.csv"
combined_5m.to_csv(out_5m, index=False)
print(f"[main] 5M merged -> {out_5m.name} ({len(combined_5m)} bars)")
print(f"  Range: {combined_5m[ts_col_5m].iloc[0]} -> {combined_5m[ts_col_5m].iloc[-1]}")

# Append to existing 1H
existing_1h = pd.read_csv(RAW_DIR / "XAUUSD_1H_2019_2025.csv")
new_1h = pd.read_csv(new_1h_path)
ts_col_1h = "datetime" if "datetime" in existing_1h.columns else existing_1h.columns[0]
ts_col_new1h = "UTC" if "UTC" in new_1h.columns else new_1h.columns[0]
new_1h = new_1h.rename(columns={ts_col_new1h: ts_col_1h})
combined_1h = pd.concat([existing_1h, new_1h]).drop_duplicates(subset=[ts_col_1h]).sort_values(ts_col_1h)
out_1h = RAW_DIR / "XAUUSD_1H_2019_2026.csv"
combined_1h.to_csv(out_1h, index=False)
print(f"[main] 1H merged -> {out_1h.name} ({len(combined_1h)} bars)")
print(f"  Range: {combined_1h[ts_col_1h].iloc[0]} -> {combined_1h[ts_col_1h].iloc[-1]}")

print("\n[done] New files ready:")
print(f"  {out_1h}")
print(f"  {out_5m}")
