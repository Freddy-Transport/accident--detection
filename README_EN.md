<p align="right">
  <a href="./README.md"><img src="https://img.shields.io/badge/语言-中文-555555?style=for-the-badge" alt="中文"></a>
  <a href="./README_EN.md"><img src="https://img.shields.io/badge/Language-English-0A7BBC?style=for-the-badge" alt="English"></a>
</p>

<h1 align="center">Early Traffic Accident Discovery for Road CCTV</h1>

<p align="center">
  Candidate triggering, trajectory evidence analysis, and VideoMAE accident recognition for fixed urban traffic cameras
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MMAction2-1.2.0-1F6FEB" alt="MMAction2 1.2.0">
  <img src="https://img.shields.io/badge/VideoMAE-ACCIDENT_Fine--tuned-C43E1C" alt="VideoMAE fine-tuned">
  <img src="https://img.shields.io/badge/Tracking-StrongSORT%20%7C%20IoU-2E8B57" alt="StrongSORT or IoU">
</p>

<p align="center">
  <img src="./docs/assets/accident_pipeline_demo.jpg" width="760" alt="Accident pipeline visualization">
</p>

<p align="center">
  <sub>Red boxes mark suspect evidence vehicles associated with the candidate segment. They are not box-level accident ground truth; VideoMAE makes the final event decision.</sub>
</p>

## Overview

This project targets early accident discovery in fixed urban road surveillance video. It does not address dashcam-based accident anticipation for autonomous driving. YOLO, multi-object tracking, and trajectory rules first select suspicious segments. A fine-tuned VideoMAE model running through MMAction2 then determines whether a segment contains an accident. The pipeline writes event JSON, structured trajectory evidence, and an annotated video.

The repository covers dataset auditing, MMAction2 annotation generation, VideoMAE fine-tuning and evaluation, end-to-end inference for new videos, and TorchScript/ONNX export.

## Features

- Local video and RTSP/RTMP/HTTP stream inputs.
- Frame-level YOLO detection JSONL for vehicles, pedestrians, and two-wheelers.
- StrongSORT tracking with an automatic lightweight IoU tracker fallback.
- Reused trajectory logic for speed drops, abnormal stops, trajectory conflicts, and queue growth.
- Binary accident classification with VideoMAE-B fine-tuned on ACCIDENT.
- Final event JSON, model scores, evidence vehicles, intermediate artifacts, and annotated video.
- Numerically validated TorchScript and ONNX VideoMAE exports.
- Dry-run event delivery by default; external requests require explicit configuration.

## Architecture

```mermaid
flowchart LR
    A[Video file or stream] --> B[YOLO detection]
    B --> C[StrongSORT / IoU tracking]
    C --> D[Trajectory anomaly rules]
    D --> E{Candidate segment?}
    E -- No --> F[Store evidence and stop]
    E -- Yes --> G[VideoMAE recognition]
    G --> H{accident_score >= threshold?}
    H -- No --> I[Store non-accident prediction]
    H -- Yes --> J[Accident event]
    C --> K[Boxes and Track IDs]
    K --> J
    J --> L[final_events.json]
    J --> M[visualization.mp4]
    J --> N[Optional event push]
```

> YOLO, tracking, and trajectory rules only generate candidates and structured evidence. The system emits an accident event only when the candidate segment reaches the VideoMAE decision threshold.

## Quick Start

### 1. Set up the environments

The validated reference environment uses Python 3.12, PyTorch 2.8.0+cu128, MMCV 2.1.0, MMEngine 0.10.7, MMAction2 1.2.0, and decord 0.6.0.

```bash
cd /root/autodl-tmp/traffic_accident_rnd
hostname
pwd
nvidia-smi || true

python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# Isolated MMAction2 environment
bash experiments/accident_mmaction2_baseline/scripts/setup_mmaction_env.sh

# ACCIDENT heuristic / Ultralytics environment
bash scripts/accident_setup_official.sh
```

Record the host, disks, Python, PyTorch, GPU, and Git revision:

```bash
bash scripts/env_check.sh
```

### 2. Provide model artifacts

Model weights and datasets are intentionally excluded from Git. The default pipeline expects these remote files:

| Purpose | Default path |
| --- | --- |
| Vehicle detector | `models/pretrained/车辆检测_v8l.pt` |
| Best VideoMAE checkpoint | `experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth` |
| TorchScript export | `models/exported/mmaction2/videomae_full/videomae_full.ts.pt` |
| ONNX export | `models/exported/mmaction2/videomae_full/videomae_full.onnx` |

Use `--yolo-model`, `--checkpoint`, and `--config` with `run_event_pipeline.py` when artifacts are stored elsewhere.

### 3. Run a new video

Store uploaded videos on the remote data disk:

```text
/autodl-fs/data/traffic_accident_rnd/user_videos/
```

Run one video:

```bash
export OMP_NUM_THREADS=1

.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video /autodl-fs/data/traffic_accident_rnd/user_videos/demo.mp4 \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5 \
  --device cuda:0
```

Run the newest uploaded video:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --latest
```

Use a stream URL:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video "rtsp://user:password@camera/stream" \
  --tracker auto \
  --device cuda:0
```

Show all options:

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --help
```

## Inference Artifacts

Each run creates an isolated directory:

```text
experiments/accident_mmaction2_baseline/outputs/YYYYMMDD_HHMM_event_pipeline_<video_id>/
```

| File | Description |
| --- | --- |
| `final_events.json` | Final accident events whose VideoMAE scores reach the threshold |
| `visualization.mp4` | Accident banner, red evidence boxes, Track IDs, and model score |
| `accident_evidence_tracks.json` | Frame-level evidence vehicle rows used by the renderer |
| `videomae_predictions.json` | Candidate accident scores and decisions |
| `candidate_segments.jsonl` | Candidate windows, trigger reasons, and evidence Track IDs |
| `trajectory_events.json` | Trajectory anomaly events |
| `detections.jsonl` / `tracks.jsonl` | YOLO detection and tracking intermediates |
| `pipeline_config.json` | Reproducible configuration snapshot |
| `logs/` | Command, duration, return code, and output for every stage |

External event delivery is disabled by default. A real request requires `--push`, `TRAFFIC_EVENT_PUSH_ENABLED=1`, and an endpoint; otherwise the pipeline writes `event_push_dry_run.json`.

## Dataset and Model

The main experiment uses the fixed-camera [ACCIDENT](https://github.com/accidentbench/ACCIDENT) dataset and derives a clip-level binary task:

- `0 non_accident`: a weak negative sampled before the event.
- `1 accident`: the event window.
- Source videos do not cross train/validation/test boundaries; group leakage is 0.
- True CCTV hard negatives are not available in the current training set and are not reported.

Full profile:

| Split | Clips | Distribution |
| --- | ---: | --- |
| Train | 625 | 219 non-accident / 406 accident |
| Validation | 153 | 52 non-accident / 101 accident |
| Test | 2468 | 948 non-accident / 1520 accident |

MMAction2 configs are under `experiments/accident_mmaction2_baseline/configs/` and cover VideoMAE, SlowFast, and X3D. The end-to-end entry point currently defaults to the full-profile VideoMAE-B checkpoint.

## Baseline Results

VideoMAE-B was initialized with Kinetics pretraining and fine-tuned for three epochs on the ACCIDENT full profile. The default deployment threshold, `0.56`, was selected on the validation split for a high-recall operating point.

| Threshold | Accuracy | Accident Precision | Accident Recall | Macro F1 | FPR | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.7382 | 0.7225 | 0.9336 | 0.6848 | 0.5749 | 0.7954 | 0.8452 |
| **0.56** | **0.7403** | **0.7448** | **0.8796** | **0.7056** | **0.4831** | **0.7954** | **0.8452** |
| 0.67 | 0.7196 | 0.8141 | 0.7059 | 0.7132 | 0.2584 | 0.7954 | 0.8452 |

Average test latency was 98.395 ms per clip, or 10.163 clips/s. Because most negatives are pre-event windows, these results must not be interpreted as the field false-alarm rate.

Detailed reports: [VideoMAE full baseline](./experiments/accident_mmaction2_baseline/reports/videomae_full_baseline_report.md) · [three-model comparison](./experiments/accident_mmaction2_baseline/reports/model_comparison_report.md) · [cascade validation](./experiments/accident_mmaction2_baseline/reports/event_pipeline_fusion_report.md)

## Model Export

```bash
.venv_mmaction/bin/python \
  experiments/accident_mmaction2_baseline/scripts/export_videomae_model.py \
  --config experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py \
  --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth \
  --device cuda:0 \
  --opset 17
```

The export accepts a preprocessed `float32[1, 3, 16, 224, 224]` RGB tensor and returns `float32[1, 2]` logits. The verified maximum absolute difference is `0.0` for TorchScript and `7.15e-7` for ONNX. The exported network still requires the exact MMAction2 sampling, color-space, and normalization contract.

## Repository Layout

```text
traffic_accident_rnd/
├── configs/                         # MVP trigger, model, and service configs
├── data/                            # Manifests and remote data-disk links
├── docs/                            # Schemas, runbooks, and interface docs
├── experiments/accident_mmaction2_baseline/
│   ├── configs/                     # VideoMAE / SlowFast / X3D configs
│   ├── data/annotations/            # MMAction2 annotations and manifest
│   ├── scripts/                     # Audit, train, evaluate, cascade, export
│   ├── reports/                     # Metrics and reproducibility reports
│   └── outputs/                     # Checkpoints and generated media, ignored
├── models/                          # Pretrained and exported weights, ignored
├── scripts/                         # Environment, dataset, and MVP commands
├── src/traffic_accident_rnd/        # Trigger, model, API, and cascade logic
├── tests/                           # Unit and regression tests
└── third_party/highway_inference_legacy/
                                      # StrongSORT and trajectory adapters
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The current regression suite reports 25 passed. Before committing, also verify that `git status` does not include checkpoints, videos, or generated experiment outputs.

## Limitations

- `non_accident` is a weak pre-event class, not genuine normal or hard-negative CCTV.
- Red evidence boxes are not box-level accident labels or ground truth collision participants.
- Stream input currently runs as one task; continuous buffering, reconnect handling, and multi-camera scheduling are not implemented.
- The SlowFast checkpoint has partial lateral temporal-kernel mismatches with the local config and is not a strict apples-to-apples comparison.
- Deployment evaluation still needs congestion, temporary stops, bus stops, construction, night glare, rain, and fog hard negatives.
- Datasets, YOLO weights, VideoMAE checkpoints, exported models, and generated videos are excluded from the repository.

## Documentation

- [New video runbook](./experiments/accident_mmaction2_baseline/reports/user_video_runbook.md)
- [Dataset audit](./experiments/accident_mmaction2_baseline/reports/dataset_audit_report.md)
- [MMAction2 environment](./experiments/accident_mmaction2_baseline/reports/mmaction2_setup_report.md)
- [Model export](./experiments/accident_mmaction2_baseline/reports/model_export_report.md)
- [Data and tracking schemas](./docs/DATA_SPEC.md)
- [Official ACCIDENT reproduction](./docs/ACCIDENT_REPRO.md)

## Acknowledgements

The dataset and video understanding baseline build on [ACCIDENT](https://github.com/accidentbench/ACCIDENT) and [MMAction2](https://github.com/open-mmlab/mmaction2). Object detection uses [Ultralytics](https://github.com/ultralytics/ultralytics). The pluggable tracking interface is informed by engineering patterns in [BoxMOT](https://github.com/mikel-brostrom/boxmot). The README structure also follows the quick-start, scope, benchmark, and modular-documentation conventions used by these widely adopted open-source projects.
