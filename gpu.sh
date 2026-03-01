#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# gpu.sh
# 目标：在 GPU 节点创建/复用 venv(python>=3.9，优先 3.10)，并从 wheelhouse 离线安装依赖。
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"

pick_python() {
  # 1) 明确指定
  if [[ -n "${P0_PYTHON_RUN:-}" ]] && command -v "${P0_PYTHON_RUN}" >/dev/null 2>&1; then
    echo "${P0_PYTHON_RUN}"; return 0
  fi
  # 2) 常见候选
  for c in python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      "$c" - <<'PY' >/dev/null 2>&1
import sys
ok = (sys.version_info.major, sys.version_info.minor) >= (3, 9)
raise SystemExit(0 if ok else 1)
PY
      if [[ $? -eq 0 ]]; then
        echo "$c"; return 0
      fi
    fi
  done
  return 1
}

PY_BIN="$(pick_python || true)"
if [[ -z "${PY_BIN}" ]]; then
  echo "[P0][gpu] ERROR: 找不到 python>=3.9（优先 3.10）。" >&2
  echo "[P0][gpu] 你现在的 python 太旧；切换到带 python3.10 的环境后再跑 gpu.sh。" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/venv"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][gpu] PYTHON     : $(command -v "${PY_BIN}")"
"${PY_BIN}" -c 'import sys; print(f"[P0][gpu] PY_VERSION  : {sys.version.split()[0]}")'

echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"
if [[ ! -d "${WHEELHOUSE}" ]] || [[ -z "$(ls -1 "${WHEELHOUSE}"/*.whl 2>/dev/null || true)" ]]; then
  echo "[P0][gpu] ERROR: wheelhouse 为空或不存在：${WHEELHOUSE}" >&2
  echo "[P0][gpu] 先在能联网的机器上运行：bash cpu.sh" >&2
  exit 1
fi

# 创建 venv：保留 --system-site-packages，方便复用系统的 torch/cuDNN 等
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][gpu] Creating venv (with --system-site-packages)..."
  "${PY_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -c 'import sys; print("[P0][gpu] VENV_PY    :", sys.executable)'

# 离线安装
set -x
python -m pip install --no-input --no-index \
  --find-links "${WHEELHOUSE}" \
  -r "${REQ_FILE}" \
  -c "${CONSTRAINT_FILE}"
set +x

# sanity
python - <<'PY'
import importlib
for m in ["pycocotools", "transformers", "qwen_vl_utils"]:
    try:
        importlib.import_module(m)
        print(f"[P0][gpu] import OK: {m}")
    except Exception as e:
        print(f"[P0][gpu] import FAIL: {m}: {e}")
        raise
PY

echo "[P0][gpu] DONE. 环境安装完成。"
echo "[P0][gpu] 运行实验：bash run.sh（或按你的调度脚本调用 main.py）"
