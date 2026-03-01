#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# P0: CPU env bootstrap (py310)
# - ALWAYS re-download wheels
# - then install offline from wheelhouse
# - venv uses --system-site-packages
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
PYBIN="${VENV_DIR}/bin/python"
PIP="${PYBIN} -m pip"

WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
mkdir -p "${WHEELHOUSE}"

echo "[P0] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0] VENV_DIR   : ${VENV_DIR}"
echo "[P0] WHEELHOUSE : ${WHEELHOUSE}"

# 0) Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[P0] Creating venv (with --system-site-packages) at: ${VENV_DIR}"
  python3.10 -m venv --system-site-packages "${VENV_DIR}"
fi

# 1) Conservative pip tooling bootstrap (avoid ultra-new pins that mirrors may not have)
echo "[P0] Bootstrapping pip tooling (conservative)..."
${PIP} install -q --upgrade "pip<25" "setuptools<82" "wheel<1" || true
${PIP} --version || true

# 2) Constraints (fix known conflict: datasets 3.2.0 requires fsspec<=2024.9.0)
CONSTRAINTS_TXT="${ROOT_DIR}/constraints.cpu.txt"
cat > "${CONSTRAINTS_TXT}" << 'EOF'
fsspec<=2024.9.0
EOF
echo "[P0] constraints written: ${CONSTRAINTS_TXT}"

# 3) Requirements (edit as needed)
REQ_TXT="${ROOT_DIR}/requirements.cpu.txt"
cat > "${REQ_TXT}" << 'EOF'
numpy
pandas
scikit-learn
Pillow
tqdm
pycocotools
qwen-vl-utils
transformers
accelerate
datasets
EOF
echo "[P0] requirements written: ${REQ_TXT}"

# 4) ALWAYS re-download wheels
echo "[P0] Cleaning wheelhouse..."
rm -rf "${WHEELHOUSE:?}/"* || true

echo "[P0] Downloading wheels into wheelhouse (network required)..."
${PIP} download -d "${WHEELHOUSE}" -c "${CONSTRAINTS_TXT}" -r "${REQ_TXT}"

# 5) Offline install from wheelhouse
echo "[P0] Installing OFFLINE from wheelhouse..."
${PIP} install --no-index --find-links "${WHEELHOUSE}" -c "${CONSTRAINTS_TXT}" -r "${REQ_TXT}"

# 6) Sanity checks
echo "[P0] Sanity check imports..."
${PYBIN} -c "import sys; print('python', sys.version)"
${PYBIN} -c "import pycocotools; print('pycocotools OK')"
${PYBIN} -c "import datasets, fsspec; print('datasets', datasets.__version__, 'fsspec', fsspec.__version__)"
echo "[P0] DONE."
