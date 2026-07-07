# X3D Pretrained Baseline Report

Config: `experiments/accident_mmaction2_baseline/configs/x3d_pretrained_accident.py`

Artifacts:

- Train/checkpoint dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch`
- Checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch/best_acc_top1_epoch_1.pth`
- Inference dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference`
- Predictions: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference/predictions/predictions.csv`
- Metrics: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference/metrics/metrics.json`
- Confusion matrix: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference/metrics/confusion_matrix.png`
- Threshold sweep: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference/metrics/threshold_sweep.csv`

Metrics on medium test clips:

- accuracy: 0.510
- accident precision: 0.505
- accident recall: 0.973
- macro F1: 0.376
- ROC-AUC: 0.583
- PR-AUC: 0.587
- false positive rate: 0.953
- average clip latency: 59.421 ms
- GPU max memory during export: 78.884 MB
- confusion matrix `[true][pred]`: `[[7, 143], [4, 146]]`

Threshold notes:

- Best macro-F1 threshold: `0.520` with macro F1 `0.535`.
- High recall threshold: `0.500` gives accident recall `0.973` and FPR `0.953`.
- Low false-alarm threshold: `0.530` gives accident recall `0.120` and FPR `0.053`.

Caveats:

- One-epoch medium run, not final model quality.
- Negative class is derived from pre-event windows, not real hard-negative CCTV.
- YOLO/Track outputs are not used as accident labels and remain evidence/candidate inputs only.
