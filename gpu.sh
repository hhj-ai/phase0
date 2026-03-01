#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"

REQ_TXT="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINTS_TXT="${ROOT_DIR}/constraints.cpu.txt"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"

test -d "${WHEELHOUSE}" || { echo "MISSING wheelhouse: ${WHEELHOUSE} (run cpu.sh first)"; exit 2; }
test -f "${REQ_TXT}" || { echo "MISSING:${REQ_TXT}"; exit 2; }
test -f "${CONSTRAINTS_TXT}" || { echo "MISSING:${CONSTRAINTS_TXT}"; exit 2; }

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[P0][gpu] Creating venv at: ${VENV_DIR}"
  python3.10 -m venv --system-site-packages "${VENV_DIR}"
fi

PYBIN="${VENV_DIR}/bin/python"
PIP="${PYBIN} -m pip"

echo "[P0][gpu] Installing OFFLINE from wheelhouse..."
${PIP} install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_TXT}" -c "${CONSTRAINTS_TXT}"

echo "[P0][gpu] Sanity checks..."
${PYBIN} -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"
${PYBIN} -c "import pycocotools; print('pycocotools OK')"

echo "[P0][gpu] Run experiment..."
${PYBIN} "${ROOT_DIR}/main.py" probe
${PYBIN} -m torch.distributed.run --nproc_per_node=8 "${ROOT_DIR}/main.py" worker
${PYBIN} "${ROOT_DIR}/main.py" analyze
${PYBIN} "${ROOT_DIR}/main.py" summary

echo "[P0][gpu] DONE."
