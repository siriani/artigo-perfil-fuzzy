#!/usr/bin/env bash
# Downloads the IPIP-50 (Big Five / OCEAN) raw responses.
# Source: Open-Source Psychometrics Project — the same file as Kaggle tunguz/big-five-personality-test.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/raw
URL="http://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip"
FINAL="$DEST/IPIP-FFM-data-8Nov2018/data-final.csv"

mkdir -p "$DEST"
if [ -f "$FINAL" ]; then
  echo "already present: $FINAL"
  wc -l "$FINAL" || true
  exit 0
fi

echo "downloading $URL ..."
curl -fL --retry 3 -o "$DEST/ipip-ffm.zip" "$URL"

echo "extracting ..."
unzip -o "$DEST/ipip-ffm.zip" -d "$DEST" >/dev/null
rm -f "$DEST/ipip-ffm.zip"

ls -lh "$DEST/IPIP-FFM-data-8Nov2018/"
echo
echo "done -> $FINAL  (tab-separated, ~1,015,342 data rows)"
