#!/usr/bin/env bash
# Fix data quality issues:
#   - Re-download 4H in annual batches (old file has 150-day gaps)
#   - Re-download 5M in quarterly batches (old annual chunks had 10-day gaps in 2022-2024)
#   - Merge each TF into a single clean consolidated CSV

set -e
RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
download_tf() {
  local TF="$1"
  local LABEL="$2"
  shift 2
  local BATCHES=("$@")
  local TOTAL=${#BATCHES[@]}
  local FAILED=()

  echo ""
  echo "=== Downloading XAUUSD $LABEL ($TOTAL batches) ==="

  for i in "${!BATCHES[@]}"; do
    read -r FROM TO <<< "${BATCHES[$i]}"
    local N=$((i + 1))
    local FNAME="xauusd-${TF}-bid-${FROM}-${TO}.csv"
    local DEST="${RAW_DIR}/${FNAME}"

    if [ -f "$DEST" ]; then
      echo "[$N/$TOTAL] SKIP $FROM -> $TO (exists)"
      continue
    fi

    echo "[$N/$TOTAL] Downloading $LABEL $FROM -> $TO ..."
    if dukascopy-node -i xauusd -from "$FROM" -to "$TO" -t "$TF" -p bid -v -f csv -dir "$RAW_DIR" --silent 2>&1; then
      echo "  -> OK: $FNAME"
    else
      echo "  -> FAILED"
      FAILED+=("$FROM $TO")
    fi
  done

  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED[@]} $LABEL batch(es) failed:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
  else
    echo "All $LABEL batches done."
  fi
}

# ─────────────────────────────────────────
# 4H — annual batches
# ─────────────────────────────────────────
BATCHES_4H=(
  "2019-01-01 2019-12-31"
  "2020-01-01 2020-12-31"
  "2021-01-01 2021-12-31"
  "2022-01-01 2022-12-31"
  "2023-01-01 2023-12-31"
  "2024-01-01 2024-12-31"
  "2025-01-01 2025-06-30"
)
download_tf "h4" "4H" "${BATCHES_4H[@]}"

# ─────────────────────────────────────────
# 5M — quarterly batches (fixes gaps in annual chunks)
# ─────────────────────────────────────────
BATCHES_5M=(
  "2019-01-01 2019-03-31"
  "2019-04-01 2019-06-30"
  "2019-07-01 2019-09-30"
  "2019-10-01 2019-12-31"
  "2020-01-01 2020-03-31"
  "2020-04-01 2020-06-30"
  "2020-07-01 2020-09-30"
  "2020-10-01 2020-12-31"
  "2021-01-01 2021-03-31"
  "2021-04-01 2021-06-30"
  "2021-07-01 2021-09-30"
  "2021-10-01 2021-12-31"
  "2022-01-01 2022-03-31"
  "2022-04-01 2022-06-30"
  "2022-07-01 2022-09-30"
  "2022-10-01 2022-12-31"
  "2023-01-01 2023-03-31"
  "2023-04-01 2023-06-30"
  "2023-07-01 2023-09-30"
  "2023-10-01 2023-12-31"
  "2024-01-01 2024-03-31"
  "2024-04-01 2024-06-30"
  "2024-07-01 2024-09-30"
  "2024-10-01 2024-12-31"
  "2025-01-01 2025-03-31"
  "2025-04-01 2025-06-30"
)
download_tf "m5" "5M" "${BATCHES_5M[@]}"

# ─────────────────────────────────────────
# Merge into consolidated CSVs
# ─────────────────────────────────────────
echo ""
echo "=== Merging into consolidated files ==="

python - << 'PYEOF'
import pandas as pd
import glob, os

RAW = "data/raw"

def merge_tf(pattern, out_name, ts_col, unix_ms=False):
    files = sorted(glob.glob(f"{RAW}/{pattern}"))
    if not files:
        print(f"  No files found for pattern: {pattern}")
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        col = df.columns[0]
        if unix_ms:
            df[col] = pd.to_datetime(df[col], unit='ms', utc=True)
        else:
            df[col] = pd.to_datetime(df[col], utc=True)
        df = df.rename(columns={col: 'datetime'})
        dfs.append(df)
    merged = pd.concat(dfs).drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
    out = f"{RAW}/{out_name}"
    merged.to_csv(out, index=False)
    print(f"  {out_name}: {len(merged):,} rows  {merged['datetime'].iloc[0]} -> {merged['datetime'].iloc[-1]}")

    # Quick gap check
    idx = pd.to_datetime(merged['datetime'])
    diffs = idx.diff().dropna()
    big = diffs[diffs > pd.Timedelta('3d')]
    non_holiday = [ts for ts, g in big.items()
                   if not (g < pd.Timedelta('4d') and idx[ts].weekday() in [0,6])]
    if non_holiday:
        print(f"  WARNING: {len(non_holiday)} suspicious gaps remain:")
        for ts in non_holiday[:5]:
            print(f"    {idx[ts]}  gap={diffs[ts]}")
    else:
        print(f"  Gap check: OK (only holiday gaps)")

print("Merging 4H...")
merge_tf("xauusd-h4-bid-20??-??-??-20??-??-??.csv", "XAUUSD_4H_2019_2025.csv", "datetime", unix_ms=True)

print("Merging 5M...")
merge_tf("xauusd-m5-bid-20??-??-??-20??-??-??.csv", "XAUUSD_5M_2019_2025_clean.csv", "datetime", unix_ms=True)

PYEOF

echo ""
echo "Done. Check data/raw/ for consolidated files."
