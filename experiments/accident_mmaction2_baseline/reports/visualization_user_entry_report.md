# Visualization And User Video Entry Report

## Remote Context

- Project root: `/root/autodl-tmp/traffic_accident_rnd`.
- User upload root: `/autodl-fs/data/traffic_accident_rnd/user_videos`.
- Convenience symlink: `data/user_videos`.
- Fine-tuned VideoMAE checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth`.
- Default accident threshold: `0.56`.
- GPU verified during this stage: RTX 4080 SUPER, CUDA visible via `nvidia-smi`.

## What Changed

- `render_accident_video.py` now renders a red `ACCIDENT DETECTED` banner on VideoMAE-positive candidate windows.
- Evidence boxes are labelled `SUSPECT_ACCIDENT_VEHICLE id=<track_id> <class>`.
- `accident_evidence_tracks.json` records the exact frame-level boxes used in rendering.
- Evidence rows are emitted only for positive VideoMAE candidates and only for candidate `evidence_track_ids`; the renderer does not label all vehicles in a window by default.
- `run_user_video.py` provides a single entrypoint for newly uploaded videos or RTSP/HTTP streams.

## New Video Command

```bash
cd /root/autodl-tmp/traffic_accident_rnd
hostname
pwd
nvidia-smi
export OMP_NUM_THREADS=1
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video /autodl-fs/data/traffic_accident_rnd/user_videos/20260708/video.mp4 \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5 \
  --device cuda:0
```

## Verification Demo

Command:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video /autodl-fs/data/traffic_accident_rnd/ACCIDENT_mmaction_clips/full/test/-6SQSDj8cYU_00__accident.mp4 \
  --output-dir experiments/accident_mmaction2_baseline/outputs/20260708_0956_event_pipeline_visual_label_positive \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5 \
  --device cuda:0
```

Result:

- Output dir: `experiments/accident_mmaction2_baseline/outputs/20260708_0956_event_pipeline_visual_label_positive`.
- Candidate count: `1`.
- Final accident events: `1`.
- VideoMAE accident score: `0.8417834639549255`.
- Trigger reasons: `abnormal_stop`, `speed_drop`.
- Evidence track ids: `1`, `3`, `4`, `5`.
- `accident_evidence_tracks.json` rows: `68`.
- `visualization.mp4` size: `1031317` bytes.

## Output Files To Inspect

- `final_events.json`: final accident event decisions.
- `accident_evidence_tracks.json`: suspect vehicle evidence boxes used by the renderer.
- `visualization.mp4`: accident banner and suspect evidence vehicle labels.
- `event_pipeline_summary.json`: consolidated artifact paths.

## Boundary

The red boxes are evidence from YOLO/Track candidate ids. They are not box-level accident ground truth. The final accident decision is still made only by VideoMAE score and threshold.
