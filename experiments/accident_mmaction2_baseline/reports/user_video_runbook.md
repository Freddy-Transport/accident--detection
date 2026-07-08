# User Video Runbook

## Upload Path

Put new videos on the remote data disk, not the system disk:

`/autodl-fs/data/traffic_accident_rnd/user_videos/`

The project symlink `data/user_videos` points to that data-disk directory for convenience.

## Upload From Local Terminal

```bash
ssh -p 25288 root@connect.westc.seetacloud.com "mkdir -p /autodl-fs/data/traffic_accident_rnd/user_videos/$(date +%Y%m%d)"
scp -P 25288 /path/to/video.mp4 root@connect.westc.seetacloud.com:/autodl-fs/data/traffic_accident_rnd/user_videos/$(date +%Y%m%d)/
```

Do not upload videos into `/root` or the project repository directory.

## Run On Remote Server

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

If the video is inside the upload root, a relative name also works:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --video 20260708/video.mp4
```

Run the newest uploaded video:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --latest
```

## Outputs

Each run writes an independent directory under:

`experiments/accident_mmaction2_baseline/outputs/YYYYMMDD_HHMM_event_pipeline_<video_id>/`

Important files:

- `final_events.json`: final traffic accident events. An event is emitted only when VideoMAE `accident_score >= 0.56`.
- `accident_evidence_tracks.json`: frame-level suspect evidence vehicle rows used for rendering.
- `visualization.mp4`: video with accident banner and red `SUSPECT_ACCIDENT_VEHICLE` boxes.
- `event_pipeline_summary.json`: all artifact paths and event count.
- `detections.jsonl`, `tracks.jsonl`, `trajectory_events.json`, `candidate_segments.jsonl`: auxiliary evidence from YOLO/Track and trajectory triggers.

## Interpretation Boundary

YOLO and Track provide candidate evidence only. Red boxes in `visualization.mp4` mark suspect evidence vehicles from the candidate track ids; they are not box-level accident ground truth. The final accident decision is gated by the fine-tuned VideoMAE checkpoint and threshold.
