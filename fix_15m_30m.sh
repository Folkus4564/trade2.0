#!/usr/bin/env bash
RAW_DIR="data/raw"

QUARTERS=(
  "2019-01-01 2019-03-31" "2019-04-01 2019-06-30" "2019-07-01 2019-09-30" "2019-10-01 2019-12-31"
  "2020-01-01 2020-03-31" "2020-04-01 2020-06-30" "2020-07-01 2020-09-30" "2020-10-01 2020-12-31"
  "2021-01-01 2021-03-31" "2021-04-01 2021-06-30" "2021-07-01 2021-09-30" "2021-10-01 2021-12-31"
  "2022-01-01 2022-03-31" "2022-04-01 2022-06-30" "2022-07-01 2022-09-30" "2022-10-01 2022-12-31"
  "2023-01-01 2023-03-31" "2023-04-01 2023-06-30" "2023-07-01 2023-09-30" "2023-10-01 2023-12-31"
  "2024-01-01 2024-03-31" "2024-04-01 2024-06-30" "2024-07-01 2024-09-30" "2024-10-01 2024-12-31"
  "2025-01-01 2025-03-31" "2025-04-01 2025-06-30"
)

download_quarters() {
  local TF="$1" LABEL="$2"
  local TOTAL=${#QUARTERS[@]}
  echo "=== Downloading $LABEL quarterly ($TOTAL batches) ==="
  for i in "${!QUARTERS[@]}"; do
    read -r FROM TO <<< "${QUARTERS[$i]}"
    local N=$((i+1))
    local DEST="${RAW_DIR}/xauusd-${TF}-bid-${FROM}-${TO}.csv"
    [ -f "$DEST" ] && echo "[$N/$TOTAL] SKIP $FROM->$TO" && continue
    echo "[$N/$TOTAL] $LABEL $FROM -> $TO ..."
    dukascopy-node -i xauusd -from "$FROM" -to "$TO" -t "$TF" -p bid -v -f csv -dir "$RAW_DIR" --silent 2>&1
  done
  echo "Done $LABEL."
}

download_quarters "m15" "15M"
download_quarters "m30" "30M"

python - << 'PYEOF'
import pandas as pd, glob

RAW = "data/raw"

def merge(pattern, out):
    files = sorted(glob.glob(f"{RAW}/{pattern}"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        col = df.columns[0]
        df[col] = pd.to_datetime(df[col], unit='ms', utc=True)
        df = df.rename(columns={col: 'datetime'})
        dfs.append(df)
    m = pd.concat(dfs).drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
    m.to_csv(f"{RAW}/{out}", index=False)
    diffs = m['datetime'].diff().dropna()
    big = diffs[diffs > pd.Timedelta('4d')]
    print(f"{out}: {len(m):,} rows | {m['datetime'].iloc[0].date()} -> {m['datetime'].iloc[-1].date()} | suspicious gaps: {len(big)}")
    for ts, g in big.items():
        print(f"  {m['datetime'].iloc[ts]}  gap={g}")

merge("xauusd-m15-bid-20??-??-??-20??-??-??.csv", "XAUUSD_15M_2019_2025.csv")
merge("xauusd-m30-bid-20??-??-??-20??-??-??.csv", "XAUUSD_30M_2019_2025.csv")
PYEOF
