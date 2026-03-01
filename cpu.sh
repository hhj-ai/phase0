#!/usr/bin/env bash
set -euo pipefail

# =========================
# Phase0 CPU prep script
# - create venv
# - download wheels into wheelhouse (network required)
# - install deps from wheelhouse
# =========================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"

# 默认优先用官方 PyPI；如果你们内网更顺畅，可以在命令前覆写：
#   INDEX_URL=... EXTRA_INDEX_URL=... sh cpu.sh
INDEX_URL="${INDEX_URL:-https://pypi.org/simple}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-https://pip.sankuai.com/simple}"

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
  echo "[P0][cpu] ERROR: 找不到可用的 Python（期望 python>=3.9）。"
  echo "           你可以："
  echo "           1) conda activate 一个带 python3.10 的环境；或"
  echo "           2) module load python/3.10；或"
  echo "           3) PYTHON=/path/to/python3.10 sh cpu.sh"
  exit 1
fi

PY_VER="$("${SYS_PY}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
PY_MAJMIN="$("${SYS_PY}" -c "import sys; print(sys.version_info.major*100+sys.version_info.minor)")"
if [[ "${PY_MAJMIN}" -lt 309 ]]; then
  echo "[P0][cpu] ERROR: 当前 python=${SYS_PY} 版本为 ${PY_VER}，太旧（需要 >=3.9，建议 3.10）。"
  echo "           请切到 python3.10 后再运行 cpu.sh。"
  exit 1
fi

PY_TAG="$("${SYS_PY}" -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')")"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/${PY_TAG}"

# pip trusted-host：只在 index/extra 是 http:// 时需要（避免被 pip 直接忽略）
mk_trusted_args() {
  local url host
  local args=()
  for url in "${INDEX_URL}" "${EXTRA_INDEX_URL}"; do
    if [[ "${url}" == http://* ]]; then
      host="$(echo "${url}" | awk -F[/:] '{print $4}')"
      [[ -n "${host}" ]] && args+=("--trusted-host" "${host}")
    fi
  done
  # PyPI 常用 host（有些环境会被识别成不安全 http 代理，这里顺手兜底）
  args+=("--trusted-host" "pypi.org" "--trusted-host" "files.pythonhosted.org")
  echo "${args[@]}"
}
TRUSTED_ARGS=($(mk_trusted_args))

echo "[P0][cpu] ROOT_DIR       : ${ROOT_DIR}"
echo "[P0][cpu] VENV_DIR       : ${VENV_DIR}"
echo "[P0][cpu] PYTHON         : ${SYS_PY} (${PY_VER})"
echo "[P0][cpu] WHEELHOUSE     : ${WHEELHOUSE}"
echo "[P0][cpu] INDEX_URL      : ${INDEX_URL}"
echo "[P0][cpu] EXTRA_INDEX_URL: ${EXTRA_INDEX_URL}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="${INDEX_URL}"
export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"

# 1) venv
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][cpu] Creating venv at: ${VENV_DIR}"
  "${SYS_PY}" -m venv "${VENV_DIR}" --system-site-packages
fi

source "${VENV_DIR}/bin/activate"

# 2) pip 基础组件（不 pin 版本，避免“镜像里没有某个版本”的坑）
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -U pip setuptools wheel "${TRUSTED_ARGS[@]}" >/dev/null

echo "[P0][cpu] venv python: $(python -V)"
echo "[P0][cpu] venv pip   : $(python -m pip -V)"

# 3) 清理并下载 wheels
echo "[P0][cpu] Refresh wheelhouse..."
rm -rf "${WHEELHOUSE}"
mkdir -p "${WHEELHOUSE}"

REQ="${ROOT_DIR}/requirements.cpu.txt"
CON="${ROOT_DIR}/constraints.cpu.txt"

if [[ ! -f "${REQ}" ]]; then
  echo "[P0][cpu] ERROR: missing ${REQ}"
  exit 1
fi
if [[ ! -f "${CON}" ]]; then
  echo "[P0][cpu] ERROR: missing ${CON}"
  exit 1
fi

echo "[P0][cpu] Downloading wheels (network required)..."
# --only-binary=:all: 逼迫走 wheel，避免在集群上编译（tokenizers/pycocotools 这种会很痛）
python -m pip download \
  -r "${REQ}" \
  -c "${CON}" \
  -d "${WHEELHOUSE}" \
  --only-binary=:all: \
  "${TRUSTED_ARGS[@]}"

# 4) 从 wheelhouse 安装
echo "[P0][cpu] Installing from wheelhouse..."
python -m pip install \
  --no-index --find-links "${WHEELHOUSE}" \
  -r "${REQ}" -c "${CON}"

# 5) sanity
python - <<'PY'
import importlib, sys
pkgs = ["numpy","pandas","sklearn","PIL","tqdm","pycocotools","transformers","accelerate","datasets","qwen_vl_utils"]
bad=[]
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception as e:
        bad.append((p, repr(e)))
if bad:
    print("[P0][cpu] MISSING/FAIL:")
    for p,e in bad:
        print(" -", p, "=>", e)
    sys.exit(2)
print("[P0][cpu] OK: core deps import fine.")
PY

echo "[P0][cpu] Done."
