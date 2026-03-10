#!/usr/bin/env bash
# Download XAUUSD 5M data using dukascopy-node (20 batches, 2019-2025)
# Usage: bash download_node.sh

set -e

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

# 20 batches: semi-annual 2019-2022, quarterly 2023-2024, quarterly 2025
BATCHES=(
  "2019-01-01 2019-06-30"
  "2019-07-01 2019-12-31"
  "2020-01-01 2020-06-30"
  "2020-07-01 2020-12-31"
  "2021-01-01 2021-06-30"
  "2021-07-01 2021-12-31"
  "2022-01-01 2022-06-30"
  "2022-07-01 2022-12-31"
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
  "2025-07-01 2025-09-30"
  "2025-10-01 2025-12-31"
)

TOTAL=${#BATCHES[@]}
FAILED=()

echo "Downloading $TOTAL batches of XAUUSD 5M (bid) data..."
echo "Output directory: $RAW_DIR"
echo ""

for i in "${!BATCHES[@]}"; do
  read -r FROM TO <<< "${BATCHES[$i]}"
  N=$((i + 1))

  # dukascopy-node names the file: xauusd-m5-bid-{from}-{to}.csv
  FNAME="xauusd-m5-bid-${FROM}-${TO}.csv"
  DEST="${RAW_DIR}/${FNAME}"

  if [ -f "$DEST" ]; then
    echo "[$N/$TOTAL] SKIP $FROM -> $TO (exists)"
    continue
  fi

  echo "[$N/$TOTAL] Downloading $FROM -> $TO ..."

  if dukascopy-node \
      -i xauusd \
      -from "$FROM" \
      -to "$TO" \
      -t m5 \
      -p bid \
      -v \
      -f csv \
      -dir "$RAW_DIR" \
      --silent 2>&1; then
    echo "  -> OK: $FNAME"
  else
    echo "  -> FAILED: $FROM $TO"
    FAILED+=("$FROM $TO")
  fi
done

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "WARNING: ${#FAILED[@]} batch(es) failed:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
else
  echo "All $TOTAL batches completed successfully."
fi

echo ""
echo "Files in $RAW_DIR:"
ls -lh "$RAW_DIR"/xauusd-m5-bid-*.csv 2>/dev/null | awk '{print $5, $9}' || echo "(none yet)"
