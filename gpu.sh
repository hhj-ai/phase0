#!/usr/bin/env bash
set -euo pipefail

# =========================
# Phase0 GPU run script
# - create venv (inherits system torch via --system-site-packages)
# - install deps from local wheelhouse
# - run: probe -> worker -> analyze -> summary
# =========================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"

# 选 python：优先 PYTHON 环境变量，其次 python3.10/python3/python
pick_python() {
  local cand
  if [[ -n "${PYTHON:-}" ]]; then
    if command -v "${PYTHON}" >/dev/null 2>&1; then
      echo "${PYTHON}"
      return 0
    fi
  fi
  for cand in python3.10 python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

SYS_PY="$(pick_python || true)"
if [[ -z "${SYS_PY}" ]]; then
  echo "[P0][gpu] ERROR: 找不到可用的 Python。"
  exit 1
fi

# venv 不存在就创建
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][gpu] Creating venv at: ${VENV_DIR}"
  "${SYS_PY}" -m venv "${VENV_DIR}" --system-site-packages
fi

source "${VENV_DIR}/bin/activate"

# wheelhouse 按 venv 的 python 版本来选
PY_TAG="$(python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')")"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/${PY_TAG}"
REQ="${ROOT_DIR}/requirements.cpu.txt"
CON="${ROOT_DIR}/constraints.cpu.txt"

echo "[P0][gpu] ROOT_DIR   : ${ROOT_DIR}"
echo "[P0][gpu] VENV_DIR   : ${VENV_DIR}"
echo "[P0][gpu] WHEELHOUSE : ${WHEELHOUSE}"

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "[P0][gpu] ERROR: wheelhouse 不存在：${WHEELHOUSE}"
  echo "          先在有网络的机器上跑：sh cpu.sh"
  exit 1
fi

# 离线安装（不连网）
python -m pip install -U pip setuptools wheel >/dev/null 2>&1 || true
python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ}" -c "${CON}"

echo "== sanity check =="
python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
PY

echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

# 你自己的数据/路径参数在 main.py 里通过 argparse 配置：
# --model_path --coco_img_dir --coco_ann_path --result_dir --log_dir
# 这里默认读 ROOT_DIR 下的相对路径（与你 repo 的 main.py 保持一致）
python "${ROOT_DIR}/main.py" probe "$@"

# 分布式 worker：默认 8 卡；需要的话可通过 NPROC 覆写：
#   NPROC=4 sh gpu.sh
NPROC="${NPROC:-8}"
torchrun --nproc_per_node="${NPROC}" "${ROOT_DIR}/main.py" worker "$@"

python "${ROOT_DIR}/main.py" analyze "$@"
python "${ROOT_DIR}/main.py" summary "$@"

echo "[P0][gpu] Done."
