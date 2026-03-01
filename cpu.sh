#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# cpu.sh
# 目标：在“任意 Python(>=3.8)”机器上，把 Phase0 运行所需依赖全部下载成 cp310 wheels。
#       不创建/修改你的运行环境，不动代码，只产出离线 wheelhouse。
#
# 产物：offline_wheels/py310/*.whl
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"

# pip 源：优先官方 PyPI；如果你们网络更偏爱公司源，可用环境变量覆盖。
P0_INDEX_URL="${P0_INDEX_URL:-https://pypi.org/simple}"
P0_EXTRA_INDEX_URL="${P0_EXTRA_INDEX_URL:-https://pip.sankuai.com/simple}"

# 下载目标：Linux x86_64 + CPython 3.10
TARGET_PLATFORM="${P0_PLATFORM:-manylinux2014_x86_64}"
TARGET_PYVER="${P0_PYVER:-310}"
TARGET_IMPL="${P0_IMPL:-cp}"
TARGET_ABI="${P0_ABI:-cp310}"

PY_BIN="${P0_PYTHON_DOWNLOAD:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "[P0][cpu] ERROR: 找不到 python/python3。" >&2
  exit 1
fi

mkdir -p "${WHEELHOUSE}"

echo "[P0][cpu] ROOT_DIR          : ${ROOT_DIR}"
echo "[P0][cpu] WHEELHOUSE        : ${WHEELHOUSE}"
echo "[P0][cpu] PY_DOWNLOAD        : $(command -v "${PY_BIN}")"
"${PY_BIN}" -c 'import sys; print(f"[P0][cpu] PY_VERSION        : {sys.version.split()[0]}")'

echo "[P0][cpu] INDEX_URL         : ${P0_INDEX_URL}"
echo "[P0][cpu] EXTRA_INDEX_URL   : ${P0_EXTRA_INDEX_URL}"
echo "[P0][cpu] TARGET            : py${TARGET_PYVER} ${TARGET_IMPL} ${TARGET_ABI} ${TARGET_PLATFORM}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[P0][cpu] ERROR: requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

# 只清 wheelhouse，不碰代码。
rm -f "${WHEELHOUSE}"/*.whl 2>/dev/null || true

PIP_COMMON_ARGS=(
  --disable-pip-version-check
  --no-input
  --timeout 20
  --retries 5
  --index-url "${P0_INDEX_URL}"
  --extra-index-url "${P0_EXTRA_INDEX_URL}"
  --trusted-host pypi.org
  --trusted-host files.pythonhosted.org
  --trusted-host pip.sankuai.com
)

# 关键：在 py3.8 机器上也能下载 py3.10 wheels（用于 GPU 侧 py3.10 venv）
DOWNLOAD_ARGS=(
  download
  -r "${REQ_FILE}"
  -c "${CONSTRAINT_FILE}"
  -d "${WHEELHOUSE}"
  --only-binary :all:
  --platform "${TARGET_PLATFORM}"
  --python-version "${TARGET_PYVER}"
  --implementation "${TARGET_IMPL}"
  --abi "${TARGET_ABI}"
)

set -x
"${PY_BIN}" -m pip "${DOWNLOAD_ARGS[@]}" "${PIP_COMMON_ARGS[@]}"
set +x

echo "[P0][cpu] DONE. Wheelhouse ready: ${WHEELHOUSE}"
echo "[P0][cpu] Next on GPU node: bash gpu.sh"
