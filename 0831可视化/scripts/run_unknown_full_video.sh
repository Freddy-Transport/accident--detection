#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/traffic_accident_rnd
DEMO="$ROOT/0831可视化"
VIDEO=${1:?"usage: $0 FULL_VIDEO [OUTPUT_DIR]"}
VIDEO=$(readlink -f "$VIDEO")
if [[ ! -f "$VIDEO" ]]; then
  echo "video not found: $VIDEO" >&2
  exit 2
fi

cd "$ROOT"
hostname
pwd
nvidia-smi || true

VIDEO_ID=$(basename "$VIDEO")
VIDEO_ID=${VIDEO_ID%.*}
OUTPUT_DIR=${2:-"$DEMO/unknown_runs/$(date +%Y%m%d_%H%M%S)_${VIDEO_ID}"}
THRESHOLD=${THRESHOLD:-0.56}
YOLO_IMGSZ=${YOLO_IMGSZ:-640}
TRACKER=${TRACKER:-strongsort}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

if [[ -z "${MMACTION_DEVICE:-}" ]]; then
  if "$ROOT/.venv_mmaction/bin/python" -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then
    MMACTION_DEVICE=cuda:0
    YOLO_DEVICE=${YOLO_DEVICE:-0}
  else
    MMACTION_DEVICE=cpu
    YOLO_DEVICE=${YOLO_DEVICE:-cpu}
  fi
fi

mkdir -p "$OUTPUT_DIR"
"$ROOT/.venv/bin/python" \
  "$ROOT/experiments/accident_mmaction2_baseline/scripts/run_event_pipeline.py" \
  --video "$VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$DEMO/models/best_acc_top1_epoch_2.pth" \
  --config "$ROOT/experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py" \
  --threshold "$THRESHOLD" \
  --device "$MMACTION_DEVICE" \
  --tracker "$TRACKER" \
  --frame-stride 1 \
  --yolo-model "$DEMO/models/车辆检测_v8l.pt" \
  --yolo-device "$YOLO_DEVICE" \
  --yolo-imgsz "$YOLO_IMGSZ" \
  --max-segments "${MAX_SEGMENTS:-8}"

echo
echo "Annotation-free inference completed."
echo "Final events: $OUTPUT_DIR/final_events.json"
echo "Visualization: $OUTPUT_DIR/visualization.mp4"
echo "Candidate segments: $OUTPUT_DIR/candidate_segments.jsonl"
echo "Suspect vehicle evidence: $OUTPUT_DIR/accident_evidence_tracks.json"
echo "Ground-truth metadata is intentionally not accepted by this command."
