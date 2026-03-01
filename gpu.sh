#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# phase0 GPU node: offline install from wheelhouse
# - 不依赖你手动 source venv（脚本里会自己 source）
# - 但脚本无法“自动把你的当前终端”切到新环境（子进程改不了父进程的 PATH）
#   要让命令行提示符切环境，你仍然需要手动：source venv/p0_env/bin/activate
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"

# wheelhouse 由 cpu.sh 在“能联网的机器”下载好；GPU 节点这里离线安装
WHEELHOUSE="${P0_WHEELHOUSE:-${ROOT_DIR}/offline_wheels/py310}"

REQ_FILE="${ROOT_DIR}/requirements.gpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.gpu.txt"

# 允许你显式指定一个 python3.10（比如来自 conda env 或旧工程 venv）
# 用法：P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh
P0_PYTHON_RUN="${P0_PYTHON_RUN:-}"

version_ok() {
  # $1 = python executable
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3,9) else 1)
PY
}

pick_python() {
  local cand
  # 1) user override
  if [[ -n "${P0_PYTHON_RUN}" ]]; then
    echo "${P0_PYTHON_RUN}"
    return 0
  fi

  # 2) if venv already exists, prefer its python (most deterministic)
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
    return 0
  fi

  # 3) otherwise search system PATH (common cases)
  for cand in python3.10 python3.11 python3.9 python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"

if [[ ! -d "${WHEELHOUSE}" ]] || [[ -z "$(ls -1 "${WHEELHOUSE}"/*.whl 2>/dev/null || true)" ]]; then
  echo "[P0][gpu] ERROR: wheelhouse 为空或不存在：${WHEELHOUSE}" >&2
  echo "[P0][gpu] 先在能联网的机器上运行：bash cpu.sh（生成离线 wheels）" >&2
  exit 1
fi

PY_BIN="$(pick_python || true)"
if [[ -z "${PY_BIN}" ]]; then
  echo "[P0][gpu] ERROR: 没找到可用的 python 可执行文件。" >&2
  echo "[P0][gpu] 你需要一个 python>=3.9（建议 3.10）。" >&2
  echo "[P0][gpu] 如果你机器上曾经有能跑起来的旧 venv，可以这么用：" >&2
  echo "           P0_PYTHON_RUN=/path/to/old_py310/bin/python bash gpu.sh" >&2
  exit 1
fi

# 如果 PY_BIN 是 venv/python，但版本太旧（比如你之前用 python3.8 创建了 venv），这里直接给出明确提示
if [[ "${PY_BIN}" == "${VENV_DIR}/bin/python" ]] && ! version_ok "${PY_BIN}"; then
  echo "[P0][gpu] ERROR: 你当前已有的 venv 里 python 版本太旧（<3.9）。" >&2
  "${PY_BIN}" -c 'import sys; print("[P0][gpu] VENV_PY_VERSION:", sys.version.split()[0])' || true
  echo "[P0][gpu] 这通常意味着：你是在 python3.8 的机器上创建了 venv/p0_env。" >&2
  echo "[P0][gpu] 解决：删掉旧 venv 并用 python3.10 重新建：" >&2
  echo "           rm -rf ${VENV_DIR}" >&2
  echo "           P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh" >&2
  exit 1
fi

# 如果还没有 venv，就用一个 >=3.9 的 python 来创建
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! version_ok "${PY_BIN}"; then
    echo "[P0][gpu] ERROR: 找到的 python 版本仍然太旧（<3.9）。" >&2
    "${PY_BIN}" -c 'import sys; print("[P0][gpu] PY_VERSION:", sys.version.split()[0])' || true
    echo "[P0][gpu] 请切到 python3.10 后再跑；或用 P0_PYTHON_RUN 指定 python3.10 路径。" >&2
    exit 1
  fi
  mkdir -p "${ROOT_DIR}/venv"
  echo "[P0][gpu] Creating venv (with --system-site-packages) using: ${PY_BIN}"
  "${PY_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

# 激活（只对当前脚本进程生效；不会改变你当前终端的环境，这是 shell 的规则）
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -c 'import sys; print("[P0][gpu] VENV_PY      :", sys.executable); print("[P0][gpu] VENV_VERSION :", sys.version.split()[0])'

# 离线安装（强制不走网络）
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

set -x
python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  -r "${REQ_FILE}" -c "${CONSTRAINT_FILE}"
set +x

# sanity imports
python - <<'PY'
import importlib
mods = ["pycocotools", "transformers", "qwen_vl_utils"]
for m in mods:
    importlib.import_module(m)
    print(f"[P0][gpu] import OK: {m}")
PY

echo "[P0][gpu] DONE. 环境安装完成。"
echo "[P0][gpu] 进入环境（影响你当前终端）：source ${VENV_DIR}/bin/activate"
echo "[P0][gpu] 运行实验：bash run.sh（或按你的调度脚本调用 main.py）"
