#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/traffic_accident_rnd}"
ACCIDENT_REPO="${ACCIDENT_REPO:-${PROJECT_ROOT}/third_party/ACCIDENT}"
ACCIDENT_DATA_ROOT="${ACCIDENT_DATA_ROOT:-/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset}"
ACCIDENT_LOG_DIR="${ACCIDENT_LOG_DIR:-${PROJECT_ROOT}/logs/accident_official}"
ACCIDENT_OUTPUT_DIR="${ACCIDENT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/accident_official_demo}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PROJECT_PIP="${PROJECT_PIP:-${PROJECT_ROOT}/.venv/bin/pip}"
KAGGLE_TOKEN_FILE="${KAGGLE_TOKEN_FILE:-/root/.kaggle/access_token}"

preflight() {
  echo "=== required preflight: hostname ==="
  hostname
  echo "=== required preflight: pwd ==="
  pwd
  echo "=== required preflight: nvidia-smi ==="
  nvidia-smi || true
}

ensure_dirs() {
  mkdir -p "${ACCIDENT_LOG_DIR}" "${ACCIDENT_OUTPUT_DIR}" "$(dirname "${ACCIDENT_DATA_ROOT}")" "${PROJECT_ROOT}/third_party"
}

ensure_kaggle_token() {
  if [[ ! -f "${KAGGLE_TOKEN_FILE}" ]]; then
    echo "Missing Kaggle token file: ${KAGGLE_TOKEN_FILE}" >&2
    echo "Create it with chmod 600 before downloading picekl/accident." >&2
    return 1
  fi
  chmod 600 "${KAGGLE_TOKEN_FILE}"
  export KAGGLE_API_TOKEN
  KAGGLE_API_TOKEN="$(cat "${KAGGLE_TOKEN_FILE}")"
}

ensure_uv() {
  if ! "${PROJECT_ROOT}/.venv/bin/uv" --version >/dev/null 2>&1; then
    "${PROJECT_PIP}" install --no-cache-dir uv
  fi
}
