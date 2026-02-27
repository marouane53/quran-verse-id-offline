#!/usr/bin/env bash
set -euo pipefail

SRC_REPO="${1:-}"

if [[ -z "${SRC_REPO}" ]]; then
  echo "Usage: scripts/import_offline_tarteel_benchmark.sh /path/to/offline-tarteel"
  exit 1
fi

if [[ ! -d "${SRC_REPO}" ]]; then
  echo "Not a directory: ${SRC_REPO}"
  exit 1
fi

SRC_CORPUS="${SRC_REPO}/benchmark/test_corpus"
if [[ ! -d "${SRC_CORPUS}" ]]; then
  echo "Could not find benchmark corpus at: ${SRC_CORPUS}"
  echo "Did you clone https://github.com/yazinsai/offline-tarteel ?"
  exit 1
fi

DST_CORPUS="benchmark/test_corpus"
mkdir -p "${DST_CORPUS}"

echo "Copying corpus from:"
echo "  ${SRC_CORPUS}"
echo "to:"
echo "  ${DST_CORPUS}"

cp -v "${SRC_CORPUS}/manifest.json" "${DST_CORPUS}/manifest.json"
# Copy common audio extensions
shopt -s nullglob
cp -v "${SRC_CORPUS}"/*.wav "${DST_CORPUS}/" 2>/dev/null || true
cp -v "${SRC_CORPUS}"/*.mp3 "${DST_CORPUS}/" 2>/dev/null || true
cp -v "${SRC_CORPUS}"/*.m4a "${DST_CORPUS}/" 2>/dev/null || true

echo "Done."
