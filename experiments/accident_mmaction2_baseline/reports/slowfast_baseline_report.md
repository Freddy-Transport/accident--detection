# SlowFast Baseline Report

Config: `configs/slowfast_accident.py`

Artifacts:

- Train/checkpoint dir: `outputs/20260707_1210_slowfast_mmaction_smoke`
- MMAction2 test dir: `outputs/20260707_1210_slowfast_mmaction_smoke_test`
- Predictions: `experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo/predictions/predictions.csv`
- Metrics: `experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo/metrics/metrics.json`
- Confusion matrix: `experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo/metrics/confusion_matrix.png`

Metrics on 40 test clips:

- accuracy: 0.375
- accident precision: 0.273
- accident recall: 0.150
- macro F1: 0.342
- ROC-AUC: 0.253
- PR-AUC: 0.382
- average clip latency: 383.2 ms
- confusion matrix `[true][pred]`: `[[12, 8], [17, 3]]`

Run commands:

```bash
experiments/accident_mmaction2_baseline/scripts/train_mmaction_model.py --model slowfast --work-dir experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_mmaction_smoke
experiments/accident_mmaction2_baseline/scripts/test_mmaction_model.py --model slowfast --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_mmaction_smoke/best_acc_top1_epoch_1.pth --work-dir experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_mmaction_smoke_test
experiments/accident_mmaction2_baseline/scripts/export_inference_demo.py --model slowfast --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_mmaction_smoke/best_acc_top1_epoch_1.pth --split test --output-dir experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo
```

Important caveats:

- 1 epoch smoke baseline, not production performance.
- No Kinetics/VideoMAE pretrained checkpoint was downloaded in this stage.
- ACCIDENT real split has no explicit normal/hard-negative videos; `non_accident` is a weak pre-event clip label.
- YOLO/Track outputs are not used as accident labels and remain evidence/candidate inputs only.
