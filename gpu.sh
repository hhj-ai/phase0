#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Phase0 GPU script = "runtime installer + runner"
# 目的：在 GPU/worker 机器上使用 python3.10 创建 venv，并从 wheelhouse 离线安装依赖。
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
REQ_FILE="${ROOT_DIR}/requirements.gpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.gpu.txt"

PY_RUN="${P0_PYTHON_RUN:-}"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "[P0][gpu] ERROR: wheelhouse 不存在：${WHEELHOUSE}"
  echo "          请先在能上网的机器跑：bash cpu.sh"
  exit 2
fi

# ---------- locate python3.10 ----------
pick_py() {
  if [[ -n "${PY_RUN}" && -x "${PY_RUN}" ]]; then echo "${PY_RUN}"; return 0; fi
  if command -v python3.10 >/dev/null 2>&1; then echo "python3.10"; return 0; fi
  if command -v python3 >/dev/null 2>&1; then
    # 如果 python3 就是 3.10 也可以
    local v; v="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null || true)"
    if [[ "${v}" == "3.10" ]]; then echo "python3"; return 0; fi
  fi
  return 1
}

if ! PY="$(pick_py)"; then
  echo "[P0][gpu] ERROR: 找不到 python3.10。"
  echo "[P0][gpu] 解决：先切到带 python3.10 的环境（conda/module），或显式指定："
  echo "          P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh"
  exit 2
fi
echo "[P0][gpu] PY_RUN     : ${PY}"

# ---------- create venv if missing ----------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[P0][gpu] Creating venv: ${VENV_DIR}"
  mkdir -p "$(dirname "${VENV_DIR}")"
  "${PY}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
VENV_PY="${VENV_DIR}/bin/python"
echo "[P0][gpu] Using venv python: ${VENV_PY}"

# pip 基础工具：尽量稳，别追新到 25（不同环境镜像会缺）
python -m pip install -q --upgrade "pip<25" "setuptools<83" "wheel" "build"

# ---------- build wheel from sdist if exists (e.g. qwen-vl-utils) ----------
# 只在 wheelhouse 里发现 tar.gz 才做；不会联网
if ls "${WHEELHOUSE}"/qwen_vl_utils-*.tar.gz >/dev/null 2>&1; then
  echo "[P0][gpu] Found qwen-vl-utils sdist, building wheel offline..."
  python -m build --wheel --no-isolation --outdir "${WHEELHOUSE}" "${WHEELHOUSE}"/qwen_vl_utils-*.tar.gz || true
fi

# ---------- offline install ----------
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[P0][gpu] ERROR: 找不到 ${REQ_FILE}（请确保仓库里有 requirements.gpu.txt）" >&2
  exit 2
fi
if [[ ! -f "${CONSTRAINT_FILE}" ]]; then
  echo "[P0][gpu] ERROR: 找不到 ${CONSTRAINT_FILE}（请确保仓库里有 constraints.gpu.txt）" >&2
  exit 2
fi

echo "[P0][gpu] Installing from wheelhouse (offline)..."
python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}"

echo "[P0][gpu] Sanity check:"
python - <<'PY'
import torch, sys
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
PY

echo "[P0][gpu] Ready. You can now run:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python main.py probe ..."
