#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/traffic_accident_rnd
DEMO="$ROOT/0831可视化"
MMACTION_PY="$ROOT/.venv_mmaction/bin/python"
APP_PY="$ROOT/.venv/bin/python"
YOLO_PY="$ROOT/third_party/ACCIDENT/baselines/heuristic/.venv/bin/python"
CONFIG="$ROOT/experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py"
CHECKPOINT="$DEMO/models/best_acc_top1_epoch_2.pth"
THRESHOLD=${THRESHOLD:-0.56}
DEMO_DEVICE=${DEMO_DEVICE:-cpu}
YOLO_DEVICE=${YOLO_DEVICE:-cpu}
SKIP_INFERENCE=${SKIP_INFERENCE:-0}

cd "$ROOT"
hostname
pwd
nvidia-smi || true

REAL_VIDEO="$DEMO/inputs/real/-6SQSDj8cYU_00__accident.mp4"
SYN_VIDEO="$DEMO/inputs/synthetic/Town06_sideswipe_clear_26.mp4"
REAL_OUT="$DEMO/outputs/real"
SYN_OUT="$DEMO/outputs/synthetic"
mkdir -p "$REAL_OUT/predictions" "$SYN_OUT/predictions" "$DEMO/logs"

score_video() {
  local video=$1 candidates=$2 output=$3 log=$4
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MMACTION_PY" \
    "$ROOT/experiments/accident_mmaction2_baseline/scripts/score_candidates_videomae.py" \
    --video "$video" --candidates-jsonl "$candidates" --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" --output-dir "$output" \
    --threshold "$THRESHOLD" --device "$DEMO_DEVICE" |& tee "$log"
}

if [[ "$SKIP_INFERENCE" != "1" ]]; then
  score_video "$REAL_VIDEO" "$REAL_OUT/candidate_segments.jsonl" "$REAL_OUT/predictions" "$DEMO/logs/real_videomae.log"
  score_video "$SYN_VIDEO" "$SYN_OUT/candidate_segments.jsonl" "$SYN_OUT/predictions" "$DEMO/logs/synthetic_videomae.log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$YOLO_PY" \
    "$ROOT/experiments/accident_mmaction2_baseline/scripts/run_yolo_detect.py" \
    --video "$REAL_VIDEO" --model "$DEMO/models/车辆检测_v8l.pt" \
    --output-jsonl "$REAL_OUT/detections.jsonl" --frame-stride 1 \
    --imgsz 640 --device "$YOLO_DEVICE" |& tee "$DEMO/logs/real_yolo.log"

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$APP_PY" \
    "$ROOT/experiments/accident_mmaction2_baseline/scripts/run_strongsort_tracker.py" \
    --video "$REAL_VIDEO" --detections-jsonl "$REAL_OUT/detections.jsonl" \
    --output-jsonl "$REAL_OUT/tracks.jsonl" --tracker strongsort \
    |& tee "$DEMO/logs/real_strongsort.log"
fi

# Synthetic red boxes come from exact CARLA collision actor annotations.
"$APP_PY" - "$SYN_VIDEO" "$SYN_OUT/tracks.jsonl" <<'PY'
import cv2, json, sys
from pathlib import Path
video, output = Path(sys.argv[1]), Path(sys.argv[2])
cap = cv2.VideoCapture(str(video))
fps, frames = float(cap.get(cv2.CAP_PROP_FPS) or 20.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
with output.open('w', encoding='utf-8') as stream:
    for index in range(frames):
        row = {'video_id': video.stem, 'frame_index': index,
               'timestamp_sec': index / fps, 'tracks': []}
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
PY

"$APP_PY" "$DEMO/scripts/render_framewise_accident_demo.py" \
  --domain real --video "$REAL_VIDEO" --tracks-jsonl "$REAL_OUT/tracks.jsonl" \
  --prediction-json "$REAL_OUT/predictions/candidate_predictions.json" \
  --metadata-json "$DEMO/metadata/real_sample_metadata.json" \
  --output-video "$REAL_OUT/real_accident_demo_visualization.mp4" \
  --output-evidence-json "$REAL_OUT/real_accident_vehicle_evidence.json" \
  --output-contact-sheet "$REAL_OUT/real_contact_sheet.jpg" \
  |& tee "$DEMO/logs/real_render.log"

"$APP_PY" "$DEMO/scripts/render_framewise_accident_demo.py" \
  --domain synthetic --video "$SYN_VIDEO" --tracks-jsonl "$SYN_OUT/tracks.jsonl" \
  --prediction-json "$SYN_OUT/predictions/candidate_predictions.json" \
  --metadata-json "$DEMO/metadata/synthetic_sample_metadata.json" \
  --synthetic-annotation "$DEMO/metadata/Town06_sideswipe_clear_26.json" \
  --output-video "$SYN_OUT/synthetic_accident_demo_visualization.mp4" \
  --output-evidence-json "$SYN_OUT/synthetic_accident_vehicle_evidence.json" \
  --output-contact-sheet "$SYN_OUT/synthetic_contact_sheet.jpg" \
  |& tee "$DEMO/logs/synthetic_render.log"

printf 'Real demo: %s\nSynthetic demo: %s\n' \
  "$REAL_OUT/real_accident_demo_visualization.mp4" \
  "$SYN_OUT/synthetic_accident_demo_visualization.mp4"
