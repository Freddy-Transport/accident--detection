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
