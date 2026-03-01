#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/common.sh"
p0_pip_sanitize_env

p0_log "[gpu] ROOT_DIR   : $ROOT_DIR"
p0_log "[gpu] WHEELHOUSE : $ROOT_DIR/offline_wheels/py310"
p0_log "[gpu] VENV_DIR   : $ROOT_DIR/venv/p0_env"

# 0) bootstrap python and wheelhouse (both idempotent)
bash "$ROOT_DIR/tools/bootstrap_python310.sh"
bash "$ROOT_DIR/tools/build_wheelhouse.sh"

# 1) ensure venv exists and CPU deps installed (offline)
bash "$ROOT_DIR/tools/create_venv.sh" "$ROOT_DIR/requirements.cpu.txt" "$ROOT_DIR/constraints.cpu.txt"

VENV_PY="$ROOT_DIR/venv/p0_env/bin/python"
WHEELHOUSE="$ROOT_DIR/offline_wheels/py310"

# 2) Torch wheels are hosted on PyTorch index. We download them into wheelhouse once, then install offline.
TORCH_INDEX="${P0_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_REQ="$ROOT_DIR/requirements.gpu.txt"
TORCH_CON="$ROOT_DIR/constraints.gpu.txt"

# Marker to avoid repeat
MARK="$WHEELHOUSE/.torch.ok"
fp_req="$(p0_sha256 "$TORCH_REQ")"
fp_con="$(p0_sha256 "$TORCH_CON")"
FP="torch_req=$fp_req torch_con=$fp_con"

need_dl=1
if [[ -f "$MARK" ]]; then
  old="$(cat "$MARK" 2>/dev/null || true)"
  [[ "$old" == "$FP" ]] && need_dl=0
fi

if [[ "$need_dl" -eq 1 ]]; then
  p0_log "[gpu] 下载 torch/torchvision/torchaudio wheels -> wheelhouse（只做一次）"
  "$VENV_PY" -m pip download -r "$TORCH_REQ" -c "$TORCH_CON" -d "$WHEELHOUSE" \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --isolated
  echo -n "$FP" > "$MARK"
else
  p0_log "[gpu] torch wheels 已下载且指纹一致，跳过下载。"
fi

p0_log "[gpu] 从 wheelhouse 离线安装 GPU 依赖（含 torch）..."
"$VENV_PY" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$TORCH_REQ" -c "$TORCH_CON"

# 3) generate env_gpu.sh to fix LD_LIBRARY_PATH precedence for pip-shipped nvidia libs (if any)
bash "$ROOT_DIR/tools/fix_ld_library_path.sh" || true

p0_log "[gpu] DONE"
p0_log "[gpu] 推荐跑之前执行：source ./env_gpu.sh （如果生成了的话）"
p0_log "[gpu] 然后：./venv/p0_env/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
