#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Phase0 CPU script: build a reproducible wheelhouse for py310
# - Uses ONLY official PyPI by default (ignores pip config/env)
# - Downloads wheels for Linux manylinux2014 x86_64 / cp310
# - Does NOT create the final runtime venv (gpu.sh will)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"

# Downloader venv can be any python (even 3.8) because it only runs "pip download".
DL_VENV="${ROOT_DIR}/.dl_venv"

# Force official PyPI (stable). You can override if needed.
INDEX_URL="${P0_INDEX_URL:-https://pypi.org/simple}"
EXTRA_INDEX_URL="${P0_EXTRA_INDEX_URL:-}"

# Skip re-download if marker exists unless FORCE=1
FORCE="${P0_FORCE_DOWNLOAD:-0}"
MARKER="${WHEELHOUSE}/.download_complete"

echo "[P0][cpu] ROOT_DIR    : ${ROOT_DIR}"
echo "[P0][cpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][cpu] INDEX_URL  : ${INDEX_URL}"
[ -n "${EXTRA_INDEX_URL}" ] && echo "[P0][cpu] EXTRA_INDEX_URL: ${EXTRA_INDEX_URL}"

mkdir -p "${WHEELHOUSE}"

if [[ -f "${MARKER}" && "${FORCE}" != "1" ]]; then
  echo "[P0][cpu] wheelhouse already complete (${MARKER} exists). Skip download."
  echo "[P0][cpu] If you want to redownload: P0_FORCE_DOWNLOAD=1 bash cpu.sh"
  exit 0
fi

# Create downloader venv (use current python3)
if [[ ! -x "${DL_VENV}/bin/python" ]]; then
  echo "[P0][cpu] Creating downloader venv at: ${DL_VENV}"
  python3 -m venv "${DL_VENV}"
fi

# Ensure downloader pip is usable, but don't pin (mirrors often lack old pins)
"${DL_VENV}/bin/python" -m pip install --upgrade --isolated -i "${INDEX_URL}" ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} pip setuptools wheel >/dev/null

echo "[P0][cpu] requirements: ${REQ_FILE}"
echo "[P0][cpu] constraints : ${CONSTRAINT_FILE}"

echo "[P0][cpu] Cleaning wheelhouse..."
rm -rf "${WHEELHOUSE:?}/"*.whl "${WHEELHOUSE:?}/"*.tar.gz "${WHEELHOUSE:?}/"*.zip "${WHEELHOUSE:?}/"*.metadata 2>/dev/null || true
rm -f "${MARKER}" || true

echo "[P0][cpu] Downloading wheels into wheelhouse (network required)..."
# IMPORTANT: When using --python-version/--platform/... pip requires --only-binary=:all: OR --no-deps.
# We use --only-binary=:all: so deps are resolved and all artifacts are wheels. (pip doc requirement)
set -x
"${DL_VENV}/bin/python" -m pip download \
  --isolated \
  -i "${INDEX_URL}" \
  ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  --dest "${WHEELHOUSE}" \
  -r "${REQ_FILE}" \
  -c "${CONSTRAINT_FILE}"
set +x

# Some packages may be sdist-only (example: qwen-vl-utils historically publishes sdist).
# Download its sdist separately WITHOUT platform restriction so you at least have the source tarball.
if ! ls "${WHEELHOUSE}"/qwen_vl_utils-*.tar.gz >/dev/null 2>&1; then
  echo "[P0][cpu] Downloading sdist for qwen-vl-utils (sdist-only on PyPI)..."
  set -x
  "${DL_VENV}/bin/python" -m pip download \
    --isolated \
    -i "${INDEX_URL}" \
    ${EXTRA_INDEX_URL:+--extra-index-url "${EXTRA_INDEX_URL}"} \
    --no-binary=:none: \
    --dest "${WHEELHOUSE}" \
    "qwen-vl-utils==0.0.14"
  set +x
fi

# Mark as complete
date -Iseconds > "${MARKER}"
echo "[P0][cpu] DONE. Wheelhouse ready: ${WHEELHOUSE}"
