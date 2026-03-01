#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Phase0 CPU script = "downloader" only
# 目的：在能上网的机器上把 Python3.10(cp310)+manylinux2014_x86_64 的 wheels 拉到本地 wheelhouse，
#      供后续 GPU/worker 机器离线安装使用。
#
# 重要：cpu.sh 不会/也不应该创建运行用的 venv（因为你这台机器可能只有 python3.8）。
#       运行环境由 gpu.sh 使用 python3.10 创建。
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${ROOT_DIR}/requirements.cpu.txt"
CONSTRAINT_FILE="${ROOT_DIR}/constraints.cpu.txt"

WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
STAMP_DIR="${WHEELHOUSE}/.p0_stamp"
STAMP_FILE="${STAMP_DIR}/stamp.json"

# 强制用官方源（避免公司镜像缺包/HTTP/证书问题）
INDEX_URL="${P0_INDEX_URL:-https://pypi.org/simple}"
EXTRA_INDEX_URL="${P0_EXTRA_INDEX_URL:-https://pypi.org/simple}"

PY_DL="${P0_PYTHON_DL:-python3}"

echo "[P0][cpu] ROOT_DIR      : ${ROOT_DIR}"
echo "[P0][cpu] PY_DL        : ${PY_DL}"
echo "[P0][cpu] INDEX_URL    : ${INDEX_URL}"
echo "[P0][cpu] WHEELHOUSE   : ${WHEELHOUSE}"
echo "[P0][cpu] REQ_FILE     : ${REQ_FILE}"
echo "[P0][cpu] CONSTRAINT   : ${CONSTRAINT_FILE}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[P0][cpu] ERROR: 找不到 ${REQ_FILE}（请确保仓库里有 requirements.cpu.txt）" >&2
  exit 2
fi
if [[ ! -f "${CONSTRAINT_FILE}" ]]; then
  echo "[P0][cpu] ERROR: 找不到 ${CONSTRAINT_FILE}（请确保仓库里有 constraints.cpu.txt）" >&2
  exit 2
fi

# ---------- stamp to skip repeated downloads ----------
mkdir -p "${WHEELHOUSE}" "${STAMP_DIR}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else python - <<'PY'
import hashlib,sys
p=sys.argv[1]
h=hashlib.sha256()
with open(p,'rb') as f:
    for c in iter(lambda:f.read(1<<20), b''):
        h.update(c)
print(h.hexdigest())
PY
  fi
}

REQ_SHA="$(sha256_file "${REQ_FILE}")"
CON_SHA="$(sha256_file "${CONSTRAINT_FILE}")"

if [[ -f "${STAMP_FILE}" ]]; then
  old_req="$(python -c "import json;print(json.load(open('${STAMP_FILE}')).get('req_sha',''))" 2>/dev/null || true)"
  old_con="$(python -c "import json;print(json.load(open('${STAMP_FILE}')).get('con_sha',''))" 2>/dev/null || true)"
  if [[ "${old_req}" == "${REQ_SHA}" && "${old_con}" == "${CON_SHA}" ]]; then
    echo "[P0][cpu] Wheelhouse 已是最新（requirements/constraints 未变），跳过下载。"
    echo "[P0][cpu] 如需强制重下：P0_FORCE_REDOWNLOAD=1 bash cpu.sh"
    if [[ "${P0_FORCE_REDOWNLOAD:-0}" != "1" ]]; then
      exit 0
    fi
  fi
fi

if ! command -v "${PY_DL}" >/dev/null 2>&1; then
  echo "[P0][cpu] ERROR: 找不到下载用解释器 ${PY_DL}" >&2
  exit 2
fi

# ---------- downloader venv (keeps pip independent & stable) ----------
DL_VENV="${ROOT_DIR}/.venv_downloader"
if [[ ! -d "${DL_VENV}" ]]; then
  echo "[P0][cpu] Creating downloader venv at ${DL_VENV}"
  "${PY_DL}" -m venv "${DL_VENV}"
fi
# shellcheck disable=SC1091
source "${DL_VENV}/bin/activate"

python -m pip install -q --upgrade "pip<25" "setuptools<83" "wheel" "build" || true

echo "[P0][cpu] Cleaning wheelhouse..."
rm -f "${WHEELHOUSE}/"*.whl "${WHEELHOUSE}/"*.tar.gz "${WHEELHOUSE}/"*.zip 2>/dev/null || true

echo "[P0][cpu] Downloading binary wheels for cp310/manylinux2014..."
# 关键点：带 --python-version/--platform 时，pip 要求 --only-binary=:all: 或 --no-deps。
# 我们需要依赖也一起下，所以用 --only-binary=:all:。
python -m pip download \
  --isolated \
  --index-url "${INDEX_URL}" \
  --extra-index-url "${EXTRA_INDEX_URL}" \
  --only-binary=:all: \
  --prefer-binary \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  -r "${REQ_FILE}" \
  -c "${CONSTRAINT_FILE}" \
  -d "${WHEELHOUSE}"

echo "[P0][cpu] Downloading sdist for packages that might not ship wheels..."
# qwen-vl-utils 有时只有 sdist（tar.gz）。我们把它单独拉下来即可；后续在 gpu.sh 里用 build-wheel 转成 whl。
python -m pip download \
  --index-url "${INDEX_URL}" \
  --extra-index-url "${EXTRA_INDEX_URL}" \
  --no-binary qwen-vl-utils \
  "qwen-vl-utils==0.0.14" \
  -d "${WHEELHOUSE}" || true

# 写 stamp
python - <<PY
import json, time, pathlib
p=pathlib.Path("${STAMP_FILE}")
p.parent.mkdir(parents=True, exist_ok=True)
json.dump({"req_sha":"${REQ_SHA}","con_sha":"${CON_SHA}","time":int(time.time())}, open(p,"w"), indent=2)
print("[P0][cpu] stamp written:", p)
PY

echo "[P0][cpu] DONE. Wheelhouse ready:"
ls -lh "${WHEELHOUSE}" | sed -n '1,120p'
