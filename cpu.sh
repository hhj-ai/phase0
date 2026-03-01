#!/usr/bin/env bash
# Phase0 env bootstrap (CPU-friendly): download wheels for cp310 + create venv with python>=3.10.
# - Works even if your *current* python is 3.8: we use it only as a downloader.
# - The runtime venv MUST be created by python3.10+ (Transformers 5.2.0 requires >=3.10).
#
# Usage:
#   bash cpu.sh
#   # (optional) force re-download:
#   P0_FORCE_DOWNLOAD=1 bash cpu.sh
#   # (optional) explicitly point to a python3.10:
#   P0_PYTHON_RUN=/path/to/python3.10 bash cpu.sh

set -euo pipefail

# Re-exec with bash if invoked via sh
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"
DL_VENV_DIR="${ROOT_DIR}/venv/p0_dl_env"
WHEELHOUSE="${ROOT_DIR}/offline_wheels/py310"
REQ="${ROOT_DIR}/requirements.cpu.txt"
CON="${ROOT_DIR}/constraints.cpu.txt"
MANIFEST="${WHEELHOUSE}/.manifest.json"

P0_INDEX_URL="${P0_INDEX_URL:-https://pypi.org/simple}"
P0_EXTRA_INDEX_URL="${P0_EXTRA_INDEX_URL:-}"
P0_FORCE_DOWNLOAD="${P0_FORCE_DOWNLOAD:-0}"

log() { echo -e "$*"; }
die() { echo -e "$*" 1>&2; exit 2; }

hash_file() {
  # sha256 of a file; portable fallback if sha256sum missing
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    python - <<'PY' "$1"
import hashlib, sys
p=sys.argv[1].encode()
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda: f.read(1<<20), b''):
        h.update(b)
print(h.hexdigest())
PY
  fi
}

# -------- downloader python (can be old) --------
pick_downloader_python() {
  if command -v python3 >/dev/null 2>&1; then echo python3; return; fi
  if command -v python  >/dev/null 2>&1; then echo python;  return; fi
  die "[P0][cpu] ERROR: 找不到 python / python3（至少需要能创建 venv 用来下载 wheels）。"
}

# -------- runtime python (MUST be >=3.10) --------
py_ver_ok() {
  # args: /path/to/python
  "$1" - <<'PY' 2>/dev/null
import sys
ok = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
print("OK" if ok else "BAD")
PY
}

pick_runtime_python() {
  # 1) explicit
  if [[ -n "${P0_PYTHON_RUN:-}" ]]; then
    if [[ -x "${P0_PYTHON_RUN}" ]] && [[ "$(py_ver_ok "${P0_PYTHON_RUN}")" == "OK" ]]; then
      echo "${P0_PYTHON_RUN}"
      return
    else
      die "[P0][cpu] ERROR: P0_PYTHON_RUN=${P0_PYTHON_RUN} 不可用或版本<3.10。"
    fi
  fi

  # 2) in PATH
  if command -v python3.10 >/dev/null 2>&1; then
    local p
    p="$(command -v python3.10)"
    if [[ "$(py_ver_ok "$p")" == "OK" ]]; then echo "$p"; return; fi
  fi
  if command -v python3 >/dev/null 2>&1; then
    local p
    p="$(command -v python3)"
    if [[ "$(py_ver_ok "$p")" == "OK" ]]; then echo "$p"; return; fi
  fi
  if command -v python >/dev/null 2>&1; then
    local p
    p="$(command -v python)"
    if [[ "$(py_ver_ok "$p")" == "OK" ]]; then echo "$p"; return; fi
  fi

  # 3) conda base (common on clusters)
  if command -v conda >/dev/null 2>&1; then
    if conda run -n base python - <<'PY' >/dev/null 2>&1
import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)
PY
    then
      echo "conda_run:base"
      return
    fi
  fi

  # 4) search sibling venvs (cheap, bounded depth)
  local found=""
  found="$(find "${ROOT_DIR}/.." -maxdepth 6 -type f -path "*/bin/python3.10" -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]] && [[ "$(py_ver_ok "$found")" == "OK" ]]; then
    echo "$found"
    return
  fi

  die "[P0][cpu] ERROR: 找不到 python>=3.10。\n\
  - Transformers 5.2.0 要求 Python>=3.10。\n\
  - 常见解法：\n\
    1) 先切到带 python3.10 的环境（conda/module），再跑：bash cpu.sh\n\
    2) 或者显式指定：P0_PYTHON_RUN=/path/to/python3.10 bash cpu.sh\n\
    3) 如果你以前某个 venv 里有 python3.10，本脚本会在 ROOT_DIR/.. 下最多搜 6 层；你也可以手动把路径喂给 P0_PYTHON_RUN。"
}

need_download() {
  [[ "$P0_FORCE_DOWNLOAD" == "1" ]] && return 0
  [[ ! -f "$MANIFEST" ]] && return 0
  [[ ! -d "$WHEELHOUSE" ]] && return 0
  # any wheels or sdists present?
  if ! ls "$WHEELHOUSE"/*.whl "$WHEELHOUSE"/*.tar.gz >/dev/null 2>&1; then
    return 0
  fi
  local want_req want_con got_req got_con
  want_req="$(hash_file "$REQ")"
  want_con="$(hash_file "$CON")"
  got_req="$(python - <<PY 2>/dev/null
import json
p="${MANIFEST}"
try:
  d=json.load(open(p))
  print(d.get("req_sha256",""))
except Exception:
  print("")
PY
)"
  got_con="$(python - <<PY 2>/dev/null
import json
p="${MANIFEST}"
try:
  d=json.load(open(p))
  print(d.get("con_sha256",""))
except Exception:
  print("")
PY
)"
  [[ "$want_req" != "$got_req" ]] && return 0
  [[ "$want_con" != "$got_con" ]] && return 0
  return 1
}

write_manifest() {
  local req_hash con_hash
  req_hash="$(hash_file "$REQ")"
  con_hash="$(hash_file "$CON")"
  python - <<PY
import json, time, os
d = {
  "created_unix": int(time.time()),
  "index_url": "${P0_INDEX_URL}",
  "extra_index_url": "${P0_EXTRA_INDEX_URL}",
  "req_sha256": "${req_hash}",
  "con_sha256": "${con_hash}",
  "python_target": "cp310-manylinux2014_x86_64",
}
os.makedirs("${WHEELHOUSE}", exist_ok=True)
with open("${MANIFEST}", "w") as f:
  json.dump(d, f, indent=2, sort_keys=True)
print("[P0][cpu] manifest written:", "${MANIFEST}")
PY
}

download_wheels() {
  mkdir -p "$WHEELHOUSE"
  mkdir -p "$(dirname "$DL_VENV_DIR")"

  local PY_DL
  PY_DL="$(pick_downloader_python)"
  log "[P0][cpu] PY_DL      : ${PY_DL}"
  log "[P0][cpu] Creating downloader venv..."
  if [[ ! -x "${DL_VENV_DIR}/bin/python" ]]; then
    "${PY_DL}" -m venv "${DL_VENV_DIR}"
  fi
  local DL_PY="${DL_VENV_DIR}/bin/python"

  # Ensure build tooling for sdist->wheel (best-effort, but usually quick)
  "${DL_PY}" -m pip install --disable-pip-version-check --no-cache-dir --upgrade "pip<25" setuptools wheel >/dev/null || true

  # Split requirements: wheels-only first, then sdist allowlist (currently just qwen-vl-utils)
  local TMP_WHEEL_REQ="${ROOT_DIR}/.tmp_requirements.wheels.txt"
  grep -vE '^\s*qwen-vl-utils\s*==' "${REQ}" > "${TMP_WHEEL_REQ}"

  log "[P0][cpu] Downloading wheels (cp310 manylinux2014) into wheelhouse (official PyPI)..."
  local args=(download --isolated --disable-pip-version-check --no-cache-dir
    --index-url "${P0_INDEX_URL}"
    --platform manylinux2014_x86_64 --implementation cp --python-version 310 --abi cp310
    --only-binary=:all:
    -r "${TMP_WHEEL_REQ}" -c "${CON}" -d "${WHEELHOUSE}"
  )
  if [[ -n "${P0_EXTRA_INDEX_URL}" ]]; then
    args+=(--extra-index-url "${P0_EXTRA_INDEX_URL}")
  fi
  "${DL_PY}" -m pip "${args[@]}"

  # Download sdist-only package(s) with --no-deps (allowed under platform restriction), then build wheel locally
  log "[P0][cpu] Downloading sdist for qwen-vl-utils (PyPI provides sdist only), then building wheel..."
  "${DL_PY}" -m pip download --isolated --disable-pip-version-check --no-cache-dir \
    --index-url "${P0_INDEX_URL}" \
    --platform manylinux2014_x86_64 --implementation cp --python-version 310 --abi cp310 \
    --no-deps --no-binary=:all: \
    "qwen-vl-utils==0.0.14" -d "${WHEELHOUSE}"

  local SDIST
  SDIST="$(ls -1 "${WHEELHOUSE}"/qwen-vl-utils-0.0.14*.tar.gz 2>/dev/null | head -n 1 || true)"
  if [[ -z "${SDIST}" ]]; then
    die "[P0][cpu] ERROR: 没下载到 qwen-vl-utils 的 sdist（检查网络/索引）。"
  fi
  # Build wheel into wheelhouse (no build isolation => uses downloader venv's setuptools/wheel)
  "${DL_PY}" -m pip wheel --disable-pip-version-check --no-deps --no-build-isolation -w "${WHEELHOUSE}" "${SDIST}"

  rm -f "${TMP_WHEEL_REQ}"
  write_manifest
  log "[P0][cpu] Wheelhouse ready: ${WHEELHOUSE}"
}

create_and_install_runtime_venv() {
  local py_run
  py_run="$(pick_runtime_python)"

  mkdir -p "$(dirname "$VENV_DIR")"

  if [[ "$py_run" == "conda_run:base" ]]; then
    log "[P0][cpu] PY_RUN    : conda run -n base python (>=3.10)"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
      conda run -n base python -m venv --system-site-packages "${VENV_DIR}"
    fi
  else
    log "[P0][cpu] PY_RUN    : ${py_run}"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
      "${py_run}" -m venv --system-site-packages "${VENV_DIR}"
    fi
  fi

  local VPY="${VENV_DIR}/bin/python"
  log "[P0][cpu] Installing from wheelhouse into venv (no index)..."
  "${VPY}" -m pip install --disable-pip-version-check --no-index --find-links "${WHEELHOUSE}" \
    -r "${REQ}" -c "${CON}"

  log "[P0][cpu] Sanity check imports..."
  "${VPY}" - <<'PY'
import sys
print("[P0] python:", sys.version)
import numpy, PIL
print("[P0] numpy:", numpy.__version__)
print("[P0] pillow:", PIL.__version__)
import transformers
print("[P0] transformers:", transformers.__version__)
import accelerate
print("[P0] accelerate:", accelerate.__version__)
import pycocotools
print("[P0] pycocotools OK")
PY

  log ""
  log "[P0][cpu] DONE."
  log "[P0][cpu] 进入环境（在当前 shell 里生效）需要你自己 source："
  log "  source \"${VENV_DIR}/bin/activate\""
  log "[P0][cpu] 注意：你用 bash/sh 运行脚本时，它是在子进程里做事，无法自动改变你当前 shell 的环境变量。"
}

main() {
  log "[P0][cpu] ROOT_DIR  : ${ROOT_DIR}"
  log "[P0][cpu] VENV_DIR  : ${VENV_DIR}"
  log "[P0][cpu] WHEELHOUSE: ${WHEELHOUSE}"
  log "[P0][cpu] INDEX_URL : ${P0_INDEX_URL}"
  [[ -n "${P0_EXTRA_INDEX_URL}" ]] && log "[P0][cpu] EXTRA_INDEX_URL: ${P0_EXTRA_INDEX_URL}"

  if need_download; then
    download_wheels
  else
    log "[P0][cpu] Wheelhouse already OK (manifest matches). Skip downloading."
    log "          (force re-download: P0_FORCE_DOWNLOAD=1 bash cpu.sh)"
  fi

  create_and_install_runtime_venv
}

main "$@"
