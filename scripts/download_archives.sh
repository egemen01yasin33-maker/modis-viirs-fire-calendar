#!/usr/bin/env bash
# Download full FIRMS archive CSVs from the GitHub Release into data/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${FIRMS_ARCHIVE_REPO:-egemen01yasin33-maker/modis-viirs-fire-calendar}"
TAG="${FIRMS_ARCHIVE_TAG:-archive-data-v1}"
DEST="${1:-$ROOT/data}"

mkdir -p "$DEST"
echo "Downloading release $TAG from $REPO → $DEST"

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" -R "$REPO" -D "$DEST" --skip-existing
else
  echo "gh CLI not found; using curl asset URLs..."
  api="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"
  mapfile -t urls < <(curl -fsSL "$api" | python3 -c '
import json,sys
rel=json.load(sys.stdin)
for a in rel.get("assets", []):
    print(a["browser_download_url"] + "\t" + a["name"])
')
  for line in "${urls[@]}"; do
    url="${line%%$'\t'*}"
    name="${line#*$'\t'}"
    out="$DEST/$name"
    if [[ -f "$out" && -s "$out" ]]; then
      echo "SKIP $name"
      continue
    fi
    echo "GET $name"
    curl -fL --retry 3 -o "$out" "$url"
  done
fi

echo "Done. Files in $DEST:"
ls -lah "$DEST"/fire_*.csv 2>/dev/null || ls -lah "$DEST"
