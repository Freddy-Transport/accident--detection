# Model Comparison Report

## Dataset

- Dataset: ACCIDENT real CCTV/fixed traffic surveillance videos.
- Raw videos: 2027 mp4; unreadable videos: 0.
- Manifest: 3246 clip rows after deriving event-window accident clips and pre-event weak non-accident clips.
- Medium profile annotations: train=300, val=112, test=300; group leakage count=0.
- Labels: `0 non_accident`, `1 accident`; `non_accident` is a weak pre-event label, not real normal/hard negative CCTV.

## Pretrained 1-Epoch Medium Results

| model | acc | accident_precision | accident_recall | macro_f1 | ROC-AUC | PR-AUC | FPR | latency_ms | clips/s | GPU MB | best_threshold | output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VideoMAE-B | 0.577 | 0.567 | 0.647 | 0.575 | 0.587 | 0.550 | 0.493 | 78.441 | 12.749 | 603.514 | 0.500 | `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference` |
| SlowFast-R50 | 0.520 | 0.513 | 0.780 | 0.485 | 0.531 | 0.531 | 0.740 | 92.519 | 10.809 | 220.278 | 0.520 | `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference` |
| X3D-S | 0.510 | 0.505 | 0.973 | 0.376 | 0.583 | 0.587 | 0.953 | 59.421 | 16.829 | 78.884 | 0.520 | `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference` |


Confusion matrices are stored under each model `metrics/confusion_matrix.png`; raw predictions are under each model `predictions/predictions.csv` and `.json`.

## Interpretation

- VideoMAE-B has the best default-threshold balance in this 1-epoch weak-label run: accident recall 0.647 and FPR 0.493.
- SlowFast-R50 reaches higher default accident recall 0.780, but FPR is 0.740; its loaded checkpoint has lateral temporal-kernel mismatches against the local config, so this result should be treated as a partial-pretrain run.
- X3D-S is fastest at 59.4 ms/clip and has recall 0.973 at threshold 0.5, but it predicts most clips as accident; FPR is 0.953.
- Threshold sweeps show that high-recall operating points still create high false alarm rates on weak pre-event negatives, so real hard negatives are required before deployment claims.

## Recommendation

- Continue with VideoMAE-B as the main semantic classifier baseline and X3D-S as the lightweight latency candidate.
- Before longer training, add true normal and hard-negative CCTV clips; current weak negatives are too easy to leak accident-context bias and too weak for false-alarm estimates.
- Fix SlowFast config/checkpoint alignment before using it for model-quality comparison.
- Keep YOLO/Track outputs as candidate trigger/evidence only, then use video classification for accident semantics.
