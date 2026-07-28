#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../LIBERO/eval_files/auto_eval_scripts/common.sh"

ensure_libero_plus_assets() {
  local libero_home="$1"
  if [ ! -d "${libero_home}/libero/libero/assets/scenes" ]; then
    echo "[ERROR] Missing LIBERO-plus assets at: ${libero_home}/libero/libero/assets" >&2
    echo "[ERROR] Please download assets.zip from https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main" >&2
    echo "[ERROR] and unzip it into ${libero_home}/libero/libero/ so that assets/scenes exists." >&2
    exit 1
  fi
}
