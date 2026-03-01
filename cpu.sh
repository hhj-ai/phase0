#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# phase0 CPU/联网机器：下载离线 wheelhouse（供 GPU 节点离线安装）
#
# 关键点：
#   - 这个脚本只负责“下载 wheels”，不碰你系统的 pip 源配置
#   - 强制使用官方 PyPI（--isolated + --index-url https://pypi.org/simple）
#   - 下载目标固定为：cp310 + manylinux2014_x86_64（给 Python3.10 / Linux x86_64）
#
# 你本机 python 不一定要是 3.10：pip download 支持用 --python-version 指定目标版本。
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_cpu_downloader"
WHEELHOUSE="${P0_WHEELHOUSE:-${ROOT_DIR}/offline_wheels/py310}"

REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"

INDEX_URL="https://pypi.org/simple"

# 允许你显式指定一个 python 来跑 downloader venv
# 用法：P0_PYTHON_DL=/path/to/python3 bash cpu.sh
P0_PYTHON_DL="${P0_PYTHON_DL:-}"

pick_python_any() {
  local cand
  if [[ -n "${P0_PYTHON_DL}" ]]; then
    echo "${P0_PYTHON_DL}"
    return 0
  fi
  for cand in python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

PY_DL="$(pick_python_any || true)"
if [[ -z "${PY_DL}" ]]; then
  echo "[P0][cpu] ERROR: 找不到 python/python3。" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/venv"
mkdir -p "${WHEELHOUSE}"

echo "[P0][cpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][cpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][cpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][cpu] PY_DL      : ${PY_DL}"
"${PY_DL}" -c 'import sys; print("[P0][cpu] PY_DL_VER  :", sys.version.split()[0])'

# downloader venv（只用于跑 pip download；避免污染系统 pip）
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][cpu] Creating downloader venv..."
  "${PY_DL}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# 确保 pip 可用（不强行升级，避免在老 python 上踩坑）
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi

echo "[P0][cpu] requirements: ${REQ_FILE}"
echo "[P0][cpu] constraints : ${CONSTRAINT_FILE}"

echo "[P0][cpu] Cleaning wheelhouse..."
rm -f "${WHEELHOUSE}"/*.whl || true

# pip download: 固定目标为 cp310 manylinux2014_x86_64
# --isolated: 忽略用户 pip.conf，避免被 http://pip.sankuai.com 之类劫持
# 注意：不要加 --no-deps，我们需要把依赖闭包也下载下来，GPU 节点才能离线装全。
echo "[P0][cpu] Downloading wheels into wheelhouse (network required)..."
set -x
python -m pip download --isolated \
  --index-url "${INDEX_URL}" \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}" \
  -d "${WHEELHOUSE}"
set +x

echo "[P0][cpu] DONE. wheelhouse 已生成：${WHEELHOUSE}"
echo "[P0][cpu] 把整个 ${WHEELHOUSE} 目录拷到 GPU 机器同一路径，然后在 GPU 节点运行：bash gpu.sh"
