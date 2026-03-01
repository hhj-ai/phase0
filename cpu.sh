#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"

REQ_TXT="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINTS_TXT="${ROOT_DIR}/constraints.cpu.txt"

# ------------------------------
# Index settings (force HTTPS)
# ------------------------------
# Some clusters ship a global pip config pointing to http://pip.sankuai.com/simple/
# which pip treats as insecure and ignores. We override via CLI args.
INDEX_URL="${INDEX_URL:-https://pip.sankuai.com/simple}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-https://pypi.org/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pip.sankuai.com}"

echo "[P0][cpu] ROOT_DIR       : ${ROOT_DIR}"
echo "[P0][cpu] VENV_DIR       : ${VENV_DIR}"
echo "[P0][cpu] WHEELHOUSE     : ${WHEELHOUSE}"
echo "[P0][cpu] INDEX_URL      : ${INDEX_URL}"
echo "[P0][cpu] EXTRA_INDEX_URL: ${EXTRA_INDEX_URL}"

mkdir -p "${WHEELHOUSE}"

# 0) Create venv (reuse system site packages, e.g., preinstalled torch)
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[P0][cpu] Creating venv at: ${VENV_DIR}"
  python3.10 -m venv --system-site-packages "${VENV_DIR}"
fi

PYBIN="${VENV_DIR}/bin/python"
PIP="${PYBIN} -m pip"

# 1) Guard files
test -f "${REQ_TXT}" || { echo "MISSING:${REQ_TXT}"; exit 2; }
test -f "${CONSTRAINTS_TXT}" || { echo "MISSING:${CONSTRAINTS_TXT}"; exit 2; }

# 2) ALWAYS re-download wheels (only wheelhouse is cleaned)
echo "[P0][cpu] Cleaning wheelhouse..."
rm -rf "${WHEELHOUSE:?}/"* || true

echo "[P0][cpu] Downloading wheels into wheelhouse (network required)..."
${PIP} download   -d "${WHEELHOUSE}"   -r "${REQ_TXT}"   -c "${CONSTRAINTS_TXT}"   -i "${INDEX_URL}"   --extra-index-url "${EXTRA_INDEX_URL}"   --trusted-host "${PIP_TRUSTED_HOST}"   --timeout 60 --retries 10

echo "[P0][cpu] Installing offline from wheelhouse..."
${PIP} install   --no-index --find-links "${WHEELHOUSE}"   -r "${REQ_TXT}"   -c "${CONSTRAINTS_TXT}"

echo "[P0][cpu] Sanity check imports..."
${PYBIN} -c "import sys; print('python', sys.version)"
${PYBIN} -c "import numpy as np; print('numpy', np.__version__)"
${PYBIN} -c "import pycocotools; print('pycocotools OK')"

echo "[P0][cpu] DONE."
