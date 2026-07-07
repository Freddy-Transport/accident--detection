#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=accident_common.sh
source "${SCRIPT_DIR}/accident_common.sh"
preflight
ensure_dirs
cd "${PROJECT_ROOT}"

if [[ -d "${ACCIDENT_REPO}/.git" ]]; then
  git -C "${ACCIDENT_REPO}" fetch --depth 1 origin main
  git -C "${ACCIDENT_REPO}" checkout main
  git -C "${ACCIDENT_REPO}" reset --hard origin/main
else
  rm -rf "${ACCIDENT_REPO}"
  git clone --depth 1 https://github.com/accidentbench/ACCIDENT "${ACCIDENT_REPO}"
fi

git -C "${ACCIDENT_REPO}" rev-parse HEAD | tee "${ACCIDENT_LOG_DIR}/accident_upstream_commit.log"
ensure_uv
cd "${ACCIDENT_REPO}"
"${PROJECT_ROOT}/.venv/bin/uv" pip install -r dataset/requirements.txt | tee "${ACCIDENT_LOG_DIR}/uv_install_dataset_requirements.log"
cd "${ACCIDENT_REPO}/baselines/heuristic"
"${PROJECT_ROOT}/.venv/bin/uv" sync | tee "${ACCIDENT_LOG_DIR}/uv_sync_heuristic.log"

# Record help output using the heuristic venv created by uv sync.
UV_PY="${ACCIDENT_REPO}/baselines/heuristic/.venv/bin/python"
"${PROJECT_ROOT}/.venv/bin/uv" pip install --python "${UV_PY}" lap onnx onnxruntime | tee "${ACCIDENT_LOG_DIR}/uv_install_heuristic_runtime.log"
"${UV_PY}" naive.py --help > "${ACCIDENT_LOG_DIR}/naive_help.log"
"${UV_PY}" optical_flow.py --help > "${ACCIDENT_LOG_DIR}/optical_flow_help.log"
"${UV_PY}" bbox_dynamics.py --help > "${ACCIDENT_LOG_DIR}/bbox_dynamics_help.log"
"${UV_PY}" - <<'PY' | tee "${ACCIDENT_LOG_DIR}/heuristic_import_check.log"
import cv2, lap, onnx, onnxruntime, pandas, ruptures, torch, ultralytics
print("cv2", cv2.__version__)
print("pandas", pandas.__version__)
print("ruptures", ruptures.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("lap", getattr(lap, "__version__", "installed"))
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
PY
