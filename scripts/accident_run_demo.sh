#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=accident_common.sh
source "${SCRIPT_DIR}/accident_common.sh"
preflight
ensure_dirs
cd "${ACCIDENT_REPO}/baselines/heuristic"
UV_PY="${ACCIDENT_REPO}/baselines/heuristic/.venv/bin/python"
DATASET_PATH="${ACCIDENT_DATA_ROOT}"
BBOX_TIMEOUT_SEC="${ACCIDENT_BBOX_TIMEOUT_SEC:-600}"
BBOX_MODEL_PATH="${ACCIDENT_BBOX_MODEL_PATH:-${PROJECT_ROOT}/models/pretrained/车辆检测_v8l.pt}"
BBOX_MODEL_NAME="$(basename -- "${BBOX_MODEL_PATH}")"
BBOX_MODEL_STEM="${BBOX_MODEL_NAME%.*}"
BBOX_DETECTIONS_DIR="${ACCIDENT_BBOX_DETECTIONS_DIR:-${ACCIDENT_REPO}/baselines/heuristic/inference-${BBOX_MODEL_STEM}}"
BBOX_BATCH_SIZE="${ACCIDENT_BBOX_BATCH_SIZE:-2}"

"${UV_PY}" naive.py --dataset-path "${DATASET_PATH}" 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_naive.log"
"${UV_PY}" optical_flow.py --dataset-path "${DATASET_PATH}" --take 1 --n-jobs 2 --overwrite 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_optical_flow_take1.log"
set +e
if [[ -f "${BBOX_MODEL_PATH}" ]]; then
  timeout "${BBOX_TIMEOUT_SEC}" "${UV_PY}" "${PROJECT_ROOT}/scripts/accident_bbox_dynamics_export_model.py" --dataset-path "${DATASET_PATH}" --take 1 --detections-dir "${BBOX_DETECTIONS_DIR}" --model-path "${BBOX_MODEL_PATH}" --image-resolution 640 --batch-size "${BBOX_BATCH_SIZE}" --overwrite 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_bbox_dynamics_take1.log"
  BBOX_RC=${PIPESTATUS[0]}
else
  echo "Missing bbox model: ${BBOX_MODEL_PATH}" | tee "${ACCIDENT_LOG_DIR}/demo_bbox_dynamics_take1.log"
  BBOX_RC=66
fi
set -e
mkdir -p "${ACCIDENT_OUTPUT_DIR}"
cp -f output_naive.csv "${ACCIDENT_OUTPUT_DIR}/output_naive.csv" 2>/dev/null || true
cp -f output_optical_flow.csv "${ACCIDENT_OUTPUT_DIR}/output_optical_flow.csv" 2>/dev/null || true
cp -f output_bbox_dynamics.csv "${ACCIDENT_OUTPUT_DIR}/output_bbox_dynamics.csv" 2>/dev/null || true
if [[ -d "${BBOX_DETECTIONS_DIR}" ]]; then
  find "${BBOX_DETECTIONS_DIR}" -maxdepth 1 -type f -name '*.json' | sort | head -n 1 | while read -r f; do cp -f "$f" "${ACCIDENT_OUTPUT_DIR}/$(basename "$f")"; done
fi
BBOX_RC="${BBOX_RC}" BBOX_TIMEOUT_SEC="${BBOX_TIMEOUT_SEC}" BBOX_MODEL_PATH="${BBOX_MODEL_PATH}" BBOX_DETECTIONS_DIR="${BBOX_DETECTIONS_DIR}" BBOX_BATCH_SIZE="${BBOX_BATCH_SIZE}" "${UV_PY}" - <<'PY' | tee "${ACCIDENT_OUTPUT_DIR}/demo_summary.json"
import json, os
from pathlib import Path
import pandas as pd
root = Path('/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset')
out = Path('/root/autodl-tmp/traffic_accident_rnd/outputs/accident_official_demo')
meta = pd.read_csv(root / 'metadata-real.csv')
meta_index = meta.set_index('path', drop=False)
first = meta.iloc[0].to_dict()
summary = {
    'demo_video_metadata': first,
    'bbox_returncode': int(os.environ['BBOX_RC']),
    'bbox_timeout_sec': int(os.environ['BBOX_TIMEOUT_SEC']),
    'bbox_model_path': os.environ['BBOX_MODEL_PATH'],
    'bbox_detections_dir': os.environ['BBOX_DETECTIONS_DIR'],
    'bbox_batch_size': int(os.environ['BBOX_BATCH_SIZE']),
}
if summary['bbox_returncode'] == 124:
    summary['bbox_status'] = 'timeout_while_running_bbox_dynamics'
elif summary['bbox_returncode'] == 66:
    summary['bbox_status'] = 'missing_model'
elif summary['bbox_returncode'] == 0:
    summary['bbox_status'] = 'completed'
else:
    summary['bbox_status'] = 'failed'

def attach_first_prediction(name: str) -> None:
    path = out / name
    if not path.exists():
        summary[name] = {'rows': 0, 'missing': True}
        return
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict() if len(df) else None
    item = {'rows': int(len(df)), 'first_prediction': row}
    if row and 'path' in row and str(row['path']) in meta_index.index:
        truth = meta_index.loc[str(row['path'])].to_dict()
        item['matched_truth'] = {
            'path': truth.get('path'),
            'accident_time': truth.get('accident_time'),
            'center_x': truth.get('center_x'),
            'center_y': truth.get('center_y'),
            'type': truth.get('type'),
        }
        if 'accident_time' in row:
            item['accident_time_abs_error_sec'] = abs(float(row['accident_time']) - float(truth['accident_time']))
    summary[name] = item

for name in ['output_naive.csv', 'output_optical_flow.csv', 'output_bbox_dynamics.csv']:
    attach_first_prediction(name)
json_files = sorted(out.glob('*.json'))
summary['copied_detection_jsons'] = [p.name for p in json_files if p.name != 'demo_summary.json']
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
if [[ "${BBOX_RC}" -ne 0 ]]; then
  echo "bbox_dynamics did not complete; see ${ACCIDENT_LOG_DIR}/demo_bbox_dynamics_take1.log" >&2
fi
exit 0
