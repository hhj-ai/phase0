#!/usr/bin/env bash
# Phase0 env bootstrap (GPU): identical to cpu.sh for dependencies; torch/cuda comes from system-site-packages.
# Usage:
#   bash gpu.sh
#   # If you want the venv to stay activated in your CURRENT shell:
#   source venv/p0_env/bin/activate

set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

# For stability, just reuse cpu.sh (wheelhouse+venv) then do a CUDA sanity check.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${ROOT_DIR}/cpu.sh"

VENV_DIR="${ROOT_DIR}/venv/p0_env"
VPY="${VENV_DIR}/bin/python"

echo ""
echo "[P0][gpu] CUDA sanity check (best-effort):"
"${VPY}" - <<'PY'
import torch, sys
print("[P0][gpu] torch:", torch.__version__)
print("[P0][gpu] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[P0][gpu] cuda device:", torch.cuda.get_device_name(0))
PY

echo "[P0][gpu] DONE. Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
