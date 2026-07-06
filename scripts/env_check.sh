#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/traffic_accident_rnd}"
mkdir -p "${PROJECT_ROOT}/logs/env"
LOG_PATH="${PROJECT_ROOT}/logs/env/env_check_$(date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="/root/miniconda3/bin/python"
fi
{
  echo "timestamp=$(date -Is)"
  echo "project_root=${PROJECT_ROOT}"
  echo "=== hostname ==="
  hostname
  echo "=== pwd ==="
  pwd
  echo "=== nvidia-smi ==="
  nvidia-smi || true
  echo "=== df -h ==="
  df -h
  echo "=== python ==="
  "${PYTHON_BIN}" --version
  echo "=== python packages ==="
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
packages = ["torch", "torchvision", "cv2", "fastapi", "uvicorn", "pytest", "kaggle"]
for name in packages:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'present' if spec else 'missing'}")
try:
    import torch
    print("torch_version", torch.__version__)
    print("torch_cuda_available", torch.cuda.is_available())
    print("torch_cuda_version", torch.version.cuda)
    print("torch_device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch_error", repr(exc))
PY
  echo "=== git ==="
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || true
} | tee "${LOG_PATH}"
echo "env_check_log=${LOG_PATH}"
