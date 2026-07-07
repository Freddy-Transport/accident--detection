#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/root/autodl-tmp/traffic_accident_rnd"
ENV_DIR="${PROJECT_ROOT}/.venv_mmaction"
REPORT_DIR="${PROJECT_ROOT}/experiments/accident_mmaction2_baseline/reports"
mkdir -p "${REPORT_DIR}" "${PROJECT_ROOT}/third_party"
python3 -m venv --system-site-packages "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com openmim mmengine "mmcv-lite>=2.0.0rc4,<2.2.0" "mmaction2==1.2.0" decord av scikit-learn pandas pyyaml importlib_metadata
"${ENV_DIR}/bin/python" -m pip install --upgrade "setuptools>=68"
"${PROJECT_ROOT}/experiments/accident_mmaction2_baseline/scripts/patch_mmaction_package.py" | tee "${REPORT_DIR}/patch_mmaction_package_report.json"
"${ENV_DIR}/bin/python" -c "import torch, mmcv, mmengine, mmaction, decord, cv2, av; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmaction', mmaction.__version__); print('decord', decord.__version__)" | tee "${REPORT_DIR}/mmaction2_setup_report.md"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1 | tee -a "${REPORT_DIR}/mmaction2_setup_report.md"
else
  echo "ffmpeg command: missing; clip extraction uses OpenCV fallback" | tee -a "${REPORT_DIR}/mmaction2_setup_report.md"
fi
