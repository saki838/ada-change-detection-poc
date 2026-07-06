set -euo pipefail

URL="https://download.pytorch.org/models/resnet34-b627a593.pth"
EXPECTED_SHA256="b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
MIN_BYTES=50000000  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/../app/weights"
DEST="${DEST_DIR}/resnet34-b627a593.pth"

mkdir -p "${DEST_DIR}"

filesize() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null; }
sha256() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }

if [ -f "${DEST}" ]; then
  SIZE="$(filesize "${DEST}")"
  if [ "${SIZE:-0}" -ge "${MIN_BYTES}" ]; then
    echo "Weights already present: ${DEST} (${SIZE} bytes) — skipping download."
    echo "sha256=$(sha256 "${DEST}")"
    exit 0
  fi
  echo "Existing file too small (${SIZE} bytes); re-downloading."
fi

echo "Downloading ResNet34 ImageNet weights (non-China, download.pytorch.org)..."
curl -fL --retry 3 -o "${DEST}" "${URL}"

SIZE="$(filesize "${DEST}")"
if [ "${SIZE:-0}" -lt "${MIN_BYTES}" ]; then
  echo "FAIL: downloaded file is only ${SIZE} bytes (expected ~83MB). Likely an HTML error page." >&2
  head -c 200 "${DEST}" >&2 || true
  exit 1
fi

ACTUAL_SHA="$(sha256 "${DEST}")"
if [ -n "${EXPECTED_SHA256}" ] && [ "${ACTUAL_SHA}" != "${EXPECTED_SHA256}" ]; then
  echo "FAIL: sha256 mismatch." >&2
  echo "  expected: ${EXPECTED_SHA256}" >&2
  echo "  actual:   ${ACTUAL_SHA}" >&2
  exit 1
fi

echo "OK: ${DEST} (${SIZE} bytes)"
echo "sha256=${ACTUAL_SHA}"