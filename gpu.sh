#!/usr/bin/env bash

# 允许用 sh 运行：自动切回 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================
# phase0 GPU/离线机器：创建运行 venv + 离线安装依赖
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${P0_VENV_DIR:-${ROOT_DIR}/venv/p0_env}"
WHEELHOUSE="${P0_WHEELHOUSE:-${ROOT_DIR}/offline_wheels/py310}"

REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"

# 你可以显式指定 runtime python（优先级最高）
# 用法：P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh
P0_PYTHON_RUN="${P0_PYTHON_RUN:-}"

ver_ge_39() {
  # $1: python executable
  "$1" - <<'PY'
import sys
maj, mi = sys.version_info[:2]
print('OK' if (maj, mi) >= (3, 9) else 'NO')
PY
}

pick_python_run() {
  local cand

  if [[ -n "${P0_PYTHON_RUN}" ]]; then
    echo "${P0_PYTHON_RUN}"; return 0
  fi

  # 1) 如果你已经激活了某个 conda/venv，通常 "python" 才是正确版本
  for cand in python python3.10 python3.11 python3.9 python3; do
    if command -v "${cand}" >/dev/null 2>&1; then
      if [[ "$(ver_ge_39 "${cand}")" == "OK" ]]; then
        echo "${cand}"; return 0
      fi
    fi
  done

  return 1
}

# 如果已有 venv，优先用它
PY_RUN=""
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PY_RUN="${VENV_DIR}/bin/python"
fi

if [[ -n "${PY_RUN}" ]]; then
  if [[ "$(ver_ge_39 "${PY_RUN}")" != "OK" ]]; then
    echo "[P0][gpu] WARN: 发现已有 venv 但 python 版本 <3.9：${PY_RUN}" >&2
    echo "[P0][gpu]      将尝试用更高版本 python 重新创建 venv。" >&2
    rm -rf "${VENV_DIR}"
    PY_RUN=""
  fi
fi

if [[ -z "${PY_RUN}" ]]; then
  PY_RUN="$(pick_python_run || true)"
  if [[ -z "${PY_RUN}" ]]; then
    echo "[P0][gpu] ERROR: 找不到 python>=3.9（优先 3.10）。" >&2
    echo "[P0][gpu] 你当前 shell 里的 python 太旧。常见解法：" >&2
    echo "  - 先切到带 python3.10 的环境（conda/module），再跑：bash gpu.sh" >&2
    echo "  - 或者显式指定：P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh" >&2
    exit 1
  fi
fi

mkdir -p "${ROOT_DIR}/venv"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"
echo "[P0][gpu] PY_RUN     : ${PY_RUN}"
"${PY_RUN}" -c 'import sys; print("[P0][gpu] PY_VER     :", sys.version.split()[0])'

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "[P0][gpu] ERROR: 找不到 wheelhouse：${WHEELHOUSE}" >&2
  echo "[P0][gpu] 先在能联网的机器上运行：bash cpu.sh 生成 offline_wheels/py310" >&2
  exit 1
fi

# 创建 venv（可重复跑，求稳）
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][gpu] Creating venv (with --system-site-packages) at: ${VENV_DIR}"
  "${PY_RUN}" -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# 确保 pip 可用
python -m ensurepip --upgrade >/dev/null 2>&1 || true

# 离线安装（只从 wheelhouse 找）
echo "[P0][gpu] Offline installing requirements..."
python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}"

# 快速自检（不通过就立刻报）
echo "[P0][gpu] Sanity check imports..."
python - <<'PY'
import sys
pkgs = [
  ('numpy', 'numpy'),
  ('torch', 'torch'),
  ('transformers', 'transformers'),
  ('datasets', 'datasets'),
  ('PIL', 'PIL'),
  ('pycocotools', 'pycocotools')
]
print('python:', sys.version.split()[0])
for name, mod in pkgs:
  try:
    __import__(mod)
    print('OK  ', name)
  except Exception as e:
    print('MISS', name, '->', e)
    raise
PY

echo "[P0][gpu] DONE. 现在可直接运行：bash run.sh" 

# 说明：脚本里 source venv 只影响脚本自身的进程，不会改变你外面的终端提示符。
# 如果你想让当前终端也进入 venv，请手动执行：
#   source "${VENV_DIR}/bin/activate"
