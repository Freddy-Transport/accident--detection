#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=accident_common.sh
source "${SCRIPT_DIR}/accident_common.sh"
preflight
ensure_dirs
ensure_kaggle_token
cd "${ACCIDENT_REPO}"
mkdir -p "${ACCIDENT_DATA_ROOT}"
"${PROJECT_ROOT}/.venv/bin/kaggle" datasets files -d picekl/accident --csv > "${ACCIDENT_LOG_DIR}/kaggle_files_picekl_accident.csv"
bash dataset/download_dataset.sh "${ACCIDENT_DATA_ROOT}" 2>&1 | tee "${ACCIDENT_LOG_DIR}/download_dataset.log"
ln -sfn "${ACCIDENT_DATA_ROOT}" "${PROJECT_ROOT}/data/official_accident"
"${PROJECT_PYTHON}" - <<'PY' | tee "${ACCIDENT_LOG_DIR}/dataset_validation.json"
import json
from pathlib import Path
import cv2
import pandas as pd
root = Path('/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset')
meta = root / 'metadata-real.csv'
videos = root / 'real_videos'
rows = pd.read_csv(meta)
video_count = len([p for p in videos.iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv'}])
first_path = videos / rows.iloc[0]['path']
cap = cv2.VideoCapture(str(first_path))
result = {
    'dataset_root': str(root),
    'metadata_real_exists': meta.is_file(),
    'metadata_real_rows': int(len(rows)),
    'real_video_count': int(video_count),
    'first_video': str(first_path),
    'first_video_opened': bool(cap.isOpened()),
    'first_video_frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    'first_video_fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    'has_metadata_synthetic': (root / 'metadata-synthetic.csv').is_file(),
    'has_synthetic_videos': (root / 'synthetic_videos').is_dir(),
}
cap.release()
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result['metadata_real_exists'] or result['metadata_real_rows'] < 1 or result['real_video_count'] < 1 or not result['first_video_opened']:
    raise SystemExit(1)
PY
du -sh "${ACCIDENT_DATA_ROOT}" | tee "${ACCIDENT_LOG_DIR}/dataset_du.log"
