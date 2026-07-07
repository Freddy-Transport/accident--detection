# SlowFast Pretrained Baseline Report

Config: `experiments/accident_mmaction2_baseline/configs/slowfast_pretrained_accident.py`

Artifacts:

- Train/checkpoint dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch`
- Checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch/best_acc_top1_epoch_1.pth`
- Inference dir: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference`
- Predictions: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference/predictions/predictions.csv`
- Metrics: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference/metrics/metrics.json`
- Confusion matrix: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference/metrics/confusion_matrix.png`
- Threshold sweep: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference/metrics/threshold_sweep.csv`

Metrics on medium test clips:

- accuracy: 0.520
- accident precision: 0.513
- accident recall: 0.780
- macro F1: 0.485
- ROC-AUC: 0.531
- PR-AUC: 0.531
- false positive rate: 0.740
- average clip latency: 92.519 ms
- GPU max memory during export: 220.278 MB
- confusion matrix `[true][pred]`: `[[39, 111], [33, 117]]`

Threshold notes:

- Best macro-F1 threshold: `0.520` with macro F1 `0.530`.
- High recall threshold: `0.480` gives accident recall `0.960` and FPR `0.893`.
- Low false-alarm threshold: `0.560` gives accident recall `0.093` and FPR `0.067`.

Caveats:

- One-epoch medium run, not final model quality.
- Negative class is derived from pre-event windows, not real hard-negative CCTV.
- YOLO/Track outputs are not used as accident labels and remain evidence/candidate inputs only.
- Warning: Kinetics checkpoint lateral temporal-kernel weights did not fully match the local SlowFast config; classifier and mismatched lateral kernels were not loaded exactly.
