# VideoMAE Baseline Report

Config: `configs/videomae_accident.py`

Artifacts:

- Train/checkpoint dir: `outputs/20260707_1158_videomae_mmaction_smoke3`
- MMAction2 test dir: `outputs/20260707_1158_videomae_mmaction_smoke3_test2`
- Predictions: `experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo/predictions/predictions.csv`
- Metrics: `experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo/metrics/metrics.json`
- Confusion matrix: `experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo/metrics/confusion_matrix.png`

Metrics on 40 test clips:

- accuracy: 0.500
- accident precision: 0.000
- accident recall: 0.000
- macro F1: 0.333
- ROC-AUC: 0.545
- PR-AUC: 0.565
- average clip latency: 232.5 ms
- confusion matrix `[true][pred]`: `[[20, 0], [20, 0]]`

Run commands:

```bash
experiments/accident_mmaction2_baseline/scripts/train_mmaction_model.py --model videomae --work-dir experiments/accident_mmaction2_baseline/outputs/20260707_1158_videomae_mmaction_smoke3
experiments/accident_mmaction2_baseline/scripts/test_mmaction_model.py --model videomae --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_1158_videomae_mmaction_smoke3/best_acc_top1_epoch_1.pth --work-dir experiments/accident_mmaction2_baseline/outputs/20260707_1158_videomae_mmaction_smoke3_test2
experiments/accident_mmaction2_baseline/scripts/export_inference_demo.py --model videomae --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_1158_videomae_mmaction_smoke3/best_acc_top1_epoch_1.pth --split test --output-dir experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo
```

Important caveats:

- 1 epoch smoke baseline, not production performance.
- No Kinetics/VideoMAE pretrained checkpoint was downloaded in this stage.
- ACCIDENT real split has no explicit normal/hard-negative videos; `non_accident` is a weak pre-event clip label.
- YOLO/Track outputs are not used as accident labels and remain evidence/candidate inputs only.
