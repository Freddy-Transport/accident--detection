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

"${UV_PY}" naive.py --dataset-path "${DATASET_PATH}" 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_naive.log"
"${UV_PY}" optical_flow.py --dataset-path "${DATASET_PATH}" --take 1 --n-jobs 2 --overwrite 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_optical_flow_take1.log"
set +e
"${UV_PY}" bbox_dynamics.py --dataset-path "${DATASET_PATH}" --take 1 --model-path yolo11x.pt --image-resolution 640 --batch-size 2 --overwrite 2>&1 | tee "${ACCIDENT_LOG_DIR}/demo_bbox_dynamics_take1.log"
BBOX_RC=${PIPESTATUS[0]}
set -e
mkdir -p "${ACCIDENT_OUTPUT_DIR}"
cp -f output_naive.csv "${ACCIDENT_OUTPUT_DIR}/output_naive.csv" 2>/dev/null || true
cp -f output_optical_flow.csv "${ACCIDENT_OUTPUT_DIR}/output_optical_flow.csv" 2>/dev/null || true
cp -f output_bbox_dynamics.csv "${ACCIDENT_OUTPUT_DIR}/output_bbox_dynamics.csv" 2>/dev/null || true
if [[ -d inference-yolo11x ]]; then
  find inference-yolo11x -maxdepth 1 -type f -name '*.json' | sort | head -n 1 | while read -r f; do cp -f "$f" "${ACCIDENT_OUTPUT_DIR}/$(basename "$f")"; done
fi
BBOX_RC="${BBOX_RC}" "${UV_PY}" - <<'PY' | tee "${ACCIDENT_OUTPUT_DIR}/demo_summary.json"
import json, os
from pathlib import Path
import pandas as pd
root = Path('/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset')
out = Path('/root/autodl-tmp/traffic_accident_rnd/outputs/accident_official_demo')
meta = pd.read_csv(root / 'metadata-real.csv')
first = meta.iloc[0].to_dict()
summary = {'demo_video_metadata': first, 'bbox_returncode': int(os.environ['BBOX_RC'])}
for name in ['output_naive.csv', 'output_optical_flow.csv', 'output_bbox_dynamics.csv']:
    path = out / name
    if path.exists():
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict() if len(df) else None
        summary[name] = {'rows': int(len(df)), 'first_prediction': row}
        if row and 'accident_time' in row and 'accident_time' in first:
            summary[name]['accident_time_abs_error_sec'] = abs(float(row['accident_time']) - float(first['accident_time']))
    else:
        summary[name] = {'rows': 0, 'missing': True}
json_files = sorted(out.glob('*.json'))
summary['copied_detection_jsons'] = [p.name for p in json_files if p.name != 'demo_summary.json']
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
if [[ "${BBOX_RC}" -ne 0 ]]; then
  echo "bbox_dynamics failed; see ${ACCIDENT_LOG_DIR}/demo_bbox_dynamics_take1.log" >&2
  exit "${BBOX_RC}"
fi
