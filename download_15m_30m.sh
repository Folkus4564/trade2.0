#!/usr/bin/env bash
# Download XAUUSD 15M and 30M data using dukascopy-node (2019-01-01 to 2025-07-01)
# 4H already exists: xauusd-h4-bid-2019-01-01-2025-07-01.csv
# Usage: bash download_15m_30m.sh

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

# Annual batches for 15M (finer resolution needs smaller chunks)
BATCHES_15M=(
  "2019-01-01 2019-12-31"
  "2020-01-01 2020-12-31"
  "2021-01-01 2021-12-31"
  "2022-01-01 2022-12-31"
  "2023-01-01 2023-12-31"
  "2024-01-01 2024-12-31"
  "2025-01-01 2025-06-30"
)

# Annual batches for 30M (coarser, annual is fine)
BATCHES_30M=(
  "2019-01-01 2019-12-31"
  "2020-01-01 2020-12-31"
  "2021-01-01 2021-12-31"
  "2022-01-01 2022-12-31"
  "2023-01-01 2023-12-31"
  "2024-01-01 2024-12-31"
  "2025-01-01 2025-06-30"
)

download_tf() {
  local TF="$1"       # e.g. m15, m30
  local LABEL="$2"    # e.g. 15M, 30M
  local -n BATCHES=$3

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

    if dukascopy-node \
        -i xauusd \
        -from "$FROM" \
        -to "$TO" \
        -t "$TF" \
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
    echo "WARNING: ${#FAILED[@]} $LABEL batch(es) failed:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
  else
    echo "All $TOTAL $LABEL batches completed successfully."
  fi

  echo "Files:"
  ls -lh "${RAW_DIR}"/xauusd-${TF}-bid-*.csv 2>/dev/null | awk '{print $5, $9}' || echo "(none)"
}

download_tf "m15" "15M" BATCHES_15M
download_tf "m30" "30M" BATCHES_30M

echo ""
echo "=== 4H already present ==="
ls -lh "${RAW_DIR}"/xauusd-h4-bid-*.csv 2>/dev/null | awk '{print $5, $9}' || echo "(not found)"

echo ""
echo "Done."
