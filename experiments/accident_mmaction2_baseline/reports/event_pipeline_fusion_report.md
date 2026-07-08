# Event Pipeline Fusion Report

## Summary

- Remote root: `/root/autodl-tmp/traffic_accident_rnd`.
- Full VideoMAE checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth`.
- Decision threshold: `0.56`; final accident events require `accident_score >= 0.56`.
- YOLO/Track only provide candidate evidence and visualization boxes; they are not treated as accident classifiers.
- Hard negative CCTV remains out of scope for this stage.

## Cleanup

- Cleanup manifest: `experiments/accident_mmaction2_baseline/reports/cleanup_manifest_20260707_1539.json`.
- Removed old smoke/medium/sanity outputs and small/medium clip caches.
- Preserved full VideoMAE run, test predictions, reports, pretrained weights, cascade demos, and full clips.
- Deleted bytes recorded by manifest: `4604189823`.

## Integrated Pipeline

Input video/RTSP flow:

`YOLO 车辆检测_v8l.pt -> StrongSORT/IoU fallback -> trackVehicleInSeqpre legacy trajectory events -> candidate segments -> fine-tuned VideoMAE -> final_events.json -> visualization.mp4`

Main command:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_event_pipeline.py \
  --video <video_or_rtsp> \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5
```

Outputs per run:

- `detections.jsonl`
- `tracks.jsonl`
- `trajectory_events.json` / `.jsonl`
- `candidate_segments.jsonl`
- `videomae_predictions.json`
- `final_events.json`
- `visualization.mp4`
- `accident_evidence_tracks.json`
- `event_push_dry_run.json` unless real push is explicitly enabled.

## Demo Results

| Case | Video | Tracker | Candidates | Final events | Score | Reasons | Visualization |
|---|---|---|---:|---:|---:|---|---|
| positive | `-6SQSDj8cYU_00__accident.mp4` | StrongSORT | 1 | 1 | 0.8418 | `abnormal_stop`, `speed_drop` | `experiments/accident_mmaction2_baseline/outputs/20260707_1618_event_pipeline_positive/visualization.mp4` |
| negative | `-6SQSDj8cYU_00__pre_event_non_accident.mp4` | StrongSORT | 0 | 0 | n/a | n/a | `experiments/accident_mmaction2_baseline/outputs/20260707_1618_event_pipeline_negative/visualization.mp4` |

Positive event output: `experiments/accident_mmaction2_baseline/outputs/20260707_1618_event_pipeline_positive/final_events.json`.
Negative event output: `experiments/accident_mmaction2_baseline/outputs/20260707_1618_event_pipeline_negative/final_events.json`.

## Model Export

- TorchScript: `/root/autodl-tmp/traffic_accident_rnd/models/exported/mmaction2/videomae_full/videomae_full.ts.pt`.
- ONNX: `/root/autodl-tmp/traffic_accident_rnd/models/exported/mmaction2/videomae_full/videomae_full.onnx`.
- Export report: `experiments/accident_mmaction2_baseline/reports/model_export_report.md`.
- Verified input: `[1, 3, 16, 224, 224]` preprocessed RGB clip tensor.
- TorchScript max abs diff: `0.0`.
- ONNX max abs diff: `7.152557373046875e-07`.

## Limits

- Current `non_accident` is ACCIDENT pre-event weak negative, not true hard negative CCTV.
- Legacy trajectory rules are treated as candidate evidence only; final accident decision is gated by VideoMAE.
- Real event push is disabled by default; enable only with `--push`, `TRAFFIC_EVENT_PUSH_ENABLED=1`, and a configured endpoint.
- Exported `.pt`/`.onnx` require the same preprocessing contract; they are not standalone video readers.

## New Video Entry

Upload new videos to `/autodl-fs/data/traffic_accident_rnd/user_videos/`. The convenience symlink is `data/user_videos`.

Run one uploaded video:

```bash
cd /root/autodl-tmp/traffic_accident_rnd
hostname
pwd
nvidia-smi
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video /autodl-fs/data/traffic_accident_rnd/user_videos/20260708/video.mp4 \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5 \
  --device cuda:0
```

Run the newest uploaded video:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --latest
```

Visualization now adds a red `ACCIDENT DETECTED` banner on VideoMAE-positive candidate windows and labels evidence boxes as `SUSPECT_ACCIDENT_VEHICLE`. These boxes come from YOLO/Track evidence ids and are not box-level accident ground truth.

## Visualization Label Verification

Updated positive demo output: `experiments/accident_mmaction2_baseline/outputs/20260708_0956_event_pipeline_visual_label_positive`.

- Candidates: `1`.
- Final accident events: `1`.
- VideoMAE accident score: `0.8417834639549255`.
- Evidence rows: `68`.
- Visualization: `experiments/accident_mmaction2_baseline/outputs/20260708_0956_event_pipeline_visual_label_positive/visualization.mp4`.
- Evidence JSON: `experiments/accident_mmaction2_baseline/outputs/20260708_0956_event_pipeline_visual_label_positive/accident_evidence_tracks.json`.

The visualization now labels only candidate evidence tracks as `SUSPECT_ACCIDENT_VEHICLE`; if no evidence track ids exist, it does not mark every vehicle in the accident window.
