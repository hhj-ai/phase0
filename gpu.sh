#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Phase0 GPU script: create py310 venv and install from wheelhouse
# - Finds python3.10 automatically (or use P0_PYTHON_RUN=...)
# - Installs offline from offline_wheels/py310 (no network by default)
# - Builds wheels from sdists if needed (qwen-vl-utils / pycocotools fallback)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
VENV_DIR="${ROOT_DIR}/venv/p0_env"

# If you *must* allow network during install (not recommended), set P0_ALLOW_NET=1
ALLOW_NET="${P0_ALLOW_NET:-0}"

# Optional: choose torch CUDA wheel index if you do need to fetch torch online.
TORCH_INDEX_URL="${P0_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# Find a python3.10 for the runtime venv.
find_py310() {
  if [[ -n "${P0_PYTHON_RUN:-}" && -x "${P0_PYTHON_RUN}" ]]; then
    echo "${P0_PYTHON_RUN}"; return 0
  fi
  # Common names/paths
  for c in python3.10 /usr/bin/python3.10 /usr/local/bin/python3.10 /opt/conda/bin/python3.10; do
    if command -v "${c}" >/dev/null 2>&1; then
      command -v "${c}"; return 0
    fi
  done
  # If inside conda env, try its python (and verify version)
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    local v
    v="$("${CONDA_PREFIX}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ "${v}" == "3.10" ]]; then
      echo "${CONDA_PREFIX}/bin/python"; return 0
    fi
  fi
  return 1
}

PY310="$(find_py310 || true)"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"

if [[ -z "${PY310}" ]]; then
  echo "[P0][gpu] ERROR: 找不到 python3.10。"
  echo "[P0][gpu] 解决："
  echo "  1) 先切到带 python3.10 的环境（conda/module）再跑：bash gpu.sh"
  echo "  2) 或显式指定：P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh"
  exit 2
fi

echo "[P0][gpu] PY310      : ${PY310}"
"${PY310}" -V

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "[P0][gpu] ERROR: wheelhouse 不存在：${WHEELHOUSE}"
  echo "[P0][gpu] 先在能联网的机器上运行：bash cpu.sh"
  exit 2
fi

# Create runtime venv
mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][gpu] Creating venv with python3.10..."
  "${PY310}" -m venv "${VENV_DIR}"
fi

# Activate
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[P0][gpu] python in venv: $(command -v python)"
python -V

# Upgrade pip tooling from wheelhouse if present, else from PyPI only when allowed
if ls "${WHEELHOUSE}"/pip-*.whl >/dev/null 2>&1; then
  python -m pip install --no-index --find-links "${WHEELHOUSE}" --upgrade pip setuptools wheel
else
  if [[ "${ALLOW_NET}" == "1" ]]; then
    python -m pip install --upgrade -i https://pypi.org/simple pip setuptools wheel
  else
    echo "[P0][gpu] WARN: wheelhouse 缺 pip/setuptools/wheel 的 whl。建议重新跑 cpu.sh 把它们也下全。"
  fi
fi

# Install everything offline
REQ_FILE="${ROOT_DIR}/requirements.gpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.gpu.txt"

echo "[P0][gpu] Installing requirements offline..."
python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}"

# If qwen-vl-utils is sdist-only, build it into wheelhouse then install
if ! python -c "import qwen_vl_utils" >/dev/null 2>&1; then
  if ls "${WHEELHOUSE}"/qwen_vl_utils-*.tar.gz >/dev/null 2>&1; then
    echo "[P0][gpu] Building wheel for qwen-vl-utils from sdist..."
    python -m pip wheel --no-deps --no-index --find-links "${WHEELHOUSE}" -w "${WHEELHOUSE}" "${WHEELHOUSE}"/qwen_vl_utils-*.tar.gz
    python -m pip install --no-index --find-links "${WHEELHOUSE}" qwen-vl-utils==0.0.14
  fi
fi

# pycocotools may be missing as a wheel on some mirrors; build from source as fallback
if ! python -c "import pycocotools" >/dev/null 2>&1; then
  echo "[P0][gpu] pycocotools not found; trying to build wheel (requires gcc/make)..."
  if [[ "${ALLOW_NET}" == "1" ]]; then
    python -m pip wheel --no-deps -i https://pypi.org/simple -w "${WHEELHOUSE}" "pycocotools==2.0.7" || true
    python -m pip install --no-index --find-links "${WHEELHOUSE}" "pycocotools==2.0.7" || true
  else
    # offline-only: try build from any sdist already in wheelhouse
    if ls "${WHEELHOUSE}"/pycocotools-*.tar.gz >/dev/null 2>&1; then
      python -m pip wheel --no-deps --no-index --find-links "${WHEELHOUSE}" -w "${WHEELHOUSE}" "${WHEELHOUSE}"/pycocotools-*.tar.gz || true
      python -m pip install --no-index --find-links "${WHEELHOUSE}" pycocotools || true
    fi
  fi
fi

echo "[P0][gpu] Sanity checks..."
python - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "ngpu:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception as e:
    print("torch import failed:", repr(e))
try:
    import pycocotools
    print("pycocotools: OK")
except Exception as e:
    print("pycocotools import failed:", repr(e))
PY

echo "[P0][gpu] DONE. To use this env later:"
echo "  source ${VENV_DIR}/bin/activate"
