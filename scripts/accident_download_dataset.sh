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
if [[ "${ACCIDENT_FORCE_DOWNLOAD:-0}" == "1" || ! -f "${ACCIDENT_DATA_ROOT}/metadata-real.csv" || ! -d "${ACCIDENT_DATA_ROOT}/real_videos" ]]; then
  bash dataset/download_dataset.sh "${ACCIDENT_DATA_ROOT}" 2>&1 | tee "${ACCIDENT_LOG_DIR}/download_dataset.log"
else
  echo "Dataset already present at ${ACCIDENT_DATA_ROOT}; set ACCIDENT_FORCE_DOWNLOAD=1 to redownload." | tee "${ACCIDENT_LOG_DIR}/download_dataset.log"
fi
"${PROJECT_PYTHON}" - <<'NORMALIZE_PY'
import csv
from pathlib import Path
root = Path('/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset')
meta = root / 'metadata-real.csv'
backup = root / 'metadata-real.original.csv'
with meta.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = reader.fieldnames
if rows and str(rows[0].get('path', '')).startswith('real_videos/'):
    if not backup.exists():
        backup.write_text(meta.read_text(encoding='utf-8'), encoding='utf-8')
    for row in rows:
        row['path'] = Path(row['path']).name
    with meta.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('normalized metadata-real.csv path column to filenames; original saved at metadata-real.original.csv')
else:
    print('metadata-real.csv path column already baseline-compatible')
NORMALIZE_PY
ln -sfn "${ACCIDENT_DATA_ROOT}" "${PROJECT_ROOT}/data/official_accident"
"${PROJECT_PYTHON}" - <<'VALIDATE_PY' | tee "${ACCIDENT_LOG_DIR}/dataset_validation.json"
import csv
import json
from pathlib import Path
import cv2
root = Path('/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset')
meta = root / 'metadata-real.csv'
videos = root / 'real_videos'
with meta.open('r', encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))
video_count = len([p for p in videos.iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv'}])
first_rel = rows[0]['path']
first_path = root / first_rel
if not first_path.exists():
    first_path = videos / first_rel
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
    'metadata_original_backup': (root / 'metadata-real.original.csv').is_file(),
}
cap.release()
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result['metadata_real_exists'] or result['metadata_real_rows'] < 1 or result['real_video_count'] < 1 or not result['first_video_opened']:
    raise SystemExit(1)
VALIDATE_PY
du -sh "${ACCIDENT_DATA_ROOT}" | tee "${ACCIDENT_LOG_DIR}/dataset_du.log"
