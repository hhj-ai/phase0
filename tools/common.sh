#!/usr/bin/env bash
set -euo pipefail

p0_log() { echo "[P0] $*"; }
p0_die() { echo "[P0] ERROR: $*" 1>&2; exit 1; }

p0_require_file() {
  local f="$1"
  [[ -f "$f" ]] || p0_die "缺少文件：$f"
}

p0_sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    # fallback
    python - <<'PY' "$f"
import hashlib,sys
p=sys.argv[1]
h=hashlib.sha256()
with open(p,'rb') as fp:
    for ch in iter(lambda: fp.read(1<<20), b''):
        h.update(ch)
print(h.hexdigest())
PY
  fi
}
