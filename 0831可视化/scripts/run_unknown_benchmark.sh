#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/traffic_accident_rnd
DEMO="$ROOT/0831可视化"
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)_unknown_full_video"}
RUN_ROOT="$DEMO/unknown_runs/$RUN_ID"
REAL_VIDEO="$DEMO/inputs/full/real/-6SQSDj8cYU_00.mp4"
SYN_VIDEO="$DEMO/inputs/full/synthetic/Town06_sideswipe_clear_26.mp4"

cd "$ROOT"
hostname
pwd
nvidia-smi || true

if [[ ! -f "$REAL_VIDEO" || ! -f "$SYN_VIDEO" ]]; then
  echo "Full-video inputs are missing under $DEMO/inputs/full" >&2
  exit 2
fi

run_and_evaluate() {
  local domain=$1 video=$2 metadata=$3
  local output="$RUN_ROOT/$domain"
  bash "$DEMO/scripts/run_unknown_full_video.sh" "$video" "$output"

  # Labels are loaded only after final_events.json has been written.
  "$ROOT/.venv/bin/python" \
    "$ROOT/experiments/accident_mmaction2_baseline/scripts/evaluate_unknown_video_events.py" \
    --video "$video" \
    --final-events-json "$output/final_events.json" \
    --tracks-jsonl "$output/tracks.jsonl" \
    --metadata-json "$metadata" \
    --output-json "$output/post_inference_evaluation.json" \
    > "$output/post_inference_evaluation.stdout.json"
}

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/PROTOCOL.txt" <<'EOF'
INFERENCE PHASE:
  full video -> YOLO -> StrongSORT -> timestamped track rules
  -> candidate clips -> VideoMAE -> suspect trigger tracks
  No accident_time, accident region, or collision actor labels are allowed.

EVALUATION PHASE:
  Load accident_time and accident region only after final_events.json exists.
EOF

run_and_evaluate real "$REAL_VIDEO" "$DEMO/metadata/real_sample_metadata.json"
run_and_evaluate synthetic "$SYN_VIDEO" "$DEMO/metadata/synthetic_sample_metadata.json"

echo "Unknown-video benchmark completed: $RUN_ROOT"
echo "Real result: $RUN_ROOT/real/final_events.json"
echo "Synthetic result: $RUN_ROOT/synthetic/final_events.json"
echo "Post-inference evaluations are stored separately in each domain directory."
