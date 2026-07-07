# VideoMAE Pretrained Baseline Report

Config: `experiments/accident_mmaction2_baseline/configs/videomae_pretrained_accident.py`

Artifacts:

- Train/checkpoint dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch`
- Checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch/best_acc_top1_epoch_1.pth`
- Inference dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference`
- Predictions: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference/predictions/predictions.csv`
- Metrics: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference/metrics/metrics.json`
- Confusion matrix: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference/metrics/confusion_matrix.png`
- Threshold sweep: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference/metrics/threshold_sweep.csv`

Metrics on medium test clips:

- accuracy: 0.577
- accident precision: 0.567
- accident recall: 0.647
- macro F1: 0.575
- ROC-AUC: 0.587
- PR-AUC: 0.550
- false positive rate: 0.493
- average clip latency: 78.441 ms
- GPU max memory during export: 603.514 MB
- confusion matrix `[true][pred]`: `[[76, 74], [53, 97]]`

Threshold notes:

- Best macro-F1 threshold: `0.500` with macro F1 `0.575`.
- High recall threshold: `0.440` gives accident recall `0.960` and FPR `0.887`.
- Low false-alarm threshold: `0.570` gives accident recall `0.093` and FPR `0.080`.

Caveats:

- One-epoch medium run, not final model quality.
- Negative class is derived from pre-event windows, not real hard-negative CCTV.
- YOLO/Track outputs are not used as accident labels and remain evidence/candidate inputs only.
