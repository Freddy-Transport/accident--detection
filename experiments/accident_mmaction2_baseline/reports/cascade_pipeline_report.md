# YOLO/Track + VideoMAE Cascade Pipeline

## Current Design

- YOLO detects vehicles/persons and writes frame-level detection JSONL.
- Track stage writes the same JSONL frame schema with `track_id`, `center_xy`, and `speed_px_per_sec`.
- Candidate trigger emits suspicious segments with trigger reasons and evidence track ids.
- Fine-tuned VideoMAE scores only candidate clips and makes the final accident / non_accident decision.
- Render stage overlays evidence boxes for accident-positive candidate windows.

## Tracker Boundary

The remote project currently contains no uploaded production Track code. The implemented tracker is a deterministic IoU fallback for reproducible demos. StrongSORT/BoxMOT or the user's local tracker can replace it by writing the same Track JSONL schema.

## Important Constraint

YOLO and Track outputs are evidence and candidate triggers only. They are not converted into accident labels, and bounding boxes are not treated as frame-level accident ground truth.

## Demo Results

- Original test-video demo: candidate generated, VideoMAE accident score `0.4742` at threshold `0.56`, so no positive alarm video frames were rendered.
- Positive clip demo: candidate generated, VideoMAE accident score `0.8841` at threshold `0.56`; rendered `experiments/accident_mmaction2_baseline/outputs/20260707_145438_cascade_positive_demo/uC7g8msxRak_00__accident_accident_visualization.mp4` with evidence boxes.
- Detection model: `/root/autodl-tmp/traffic_accident_rnd/models/pretrained/车辆检测_v8l.pt`.
- Tracker: deterministic IoU fallback; StrongSORT/BoxMOT remains an interchangeable future backend through the same Track JSONL schema.
