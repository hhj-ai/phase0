#!/usr/bin/env bash

# 允许用 sh 运行：自动切回 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================
# phase0 CPU/联网机器：下载离线 wheelhouse（供 GPU 节点离线安装）
#
# 目标：一次下载齐全、可复现、尽量不受你机器 pip 配置影响。
#
# 核心策略（求稳版）：
#   1) 强制用官方 PyPI：--isolated + --index-url https://pypi.org/simple
#   2) 固定目标平台/解释器：manylinux2014_x86_64 + cp310
#   3) 只下二进制 wheel：--only-binary=:all:  （避免 sdist 需要编译）
#      说明：pip 在指定 --platform/--python-version 时，必须配合 --only-binary 或 --no-deps。
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
    echo "${P0_PYTHON_DL}"; return 0
  fi
  # downloader 只需要“能跑 pip”，不要求 3.10
  for cand in python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      echo "${cand}"; return 0
    fi
  done
  return 1
}

PY_DL="$(pick_python_any || true)"
if [[ -z "${PY_DL}" ]]; then
  echo "[P0][cpu] ERROR: 找不到 python/python3。" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/venv" "${WHEELHOUSE}"

echo "[P0][cpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][cpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][cpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][cpu] PY_DL      : ${PY_DL}"
"${PY_DL}" -c 'import sys; print("[P0][cpu] PY_DL_VER  :", sys.version.split()[0])'

echo "[P0][cpu] requirements: ${REQ_FILE}"
echo "[P0][cpu] constraints : ${CONSTRAINT_FILE}"

# downloader venv（只用于跑 pip download；避免污染系统 pip）
rm -rf "${VENV_DIR}"
"${PY_DL}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# 确保 pip 可用
python -m ensurepip --upgrade >/dev/null 2>&1 || true

# 清空 wheelhouse（你说“直接下”，那就每次都重新下）
echo "[P0][cpu] Cleaning wheelhouse..."
rm -f "${WHEELHOUSE}"/*.whl >/dev/null 2>&1 || true

# pip download: 固定目标为 cp310 manylinux2014_x86_64
# --isolated: 忽略用户 pip.conf / 环境变量（避免被 http://pip.sankuai.com 劫持）
# --only-binary=:all:：只拉 wheel（求稳；不让它下源码包）
# --no-binary=:none:：显式保证 no-binary 没被别的配置偷偷设置
# --no-cache-dir：避免缓存污染

echo "[P0][cpu] Downloading wheels into wheelhouse (network required)..."
set -x
python -m pip download \
  --isolated \
  --disable-pip-version-check \
  --no-cache-dir \
  --index-url "${INDEX_URL}" \
  --only-binary=:all: \
  --no-binary=:none: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}" \
  -d "${WHEELHOUSE}"
set +x

# 轻量校验：至少要看到这些关键轮子
missing=0
for key in numpy transformers tokenizers safetensors pycocotools; do
  if ! ls "${WHEELHOUSE}"/${key}-*.whl >/dev/null 2>&1; then
    echo "[P0][cpu] WARN: wheelhouse 中没看到 ${key} 的 wheel（可能是依赖图变化或版本不匹配）" >&2
    missing=1
  fi
done

if [[ "${missing}" -eq 1 ]]; then
  echo "[P0][cpu] NOTE: 上面只是提示，最终以 GPU 侧离线安装能否成功为准。" >&2
fi

echo "[P0][cpu] DONE. wheelhouse 已生成：${WHEELHOUSE}"
echo "[P0][cpu] 下一步：在 GPU 节点运行 bash gpu.sh（gpu.sh 会离线安装）。"
