# Model Comparison Report

Dataset and split:

- ACCIDENT real metadata rows: 2027
- Real videos checked: 2027
- Corrupt/unreadable videos: 0
- Original split: train=507, test=1520
- Manifest rows: 3246 (`accident` windows plus derived weak `non_accident` windows).
- Generated smoke annotations: train=40, val=40, test=40
- Class mapping: `0 non_accident`, `1 accident`; hard negative is reserved but disabled.

| model | acc | accident_precision | accident_recall | macro_f1 | roc_auc | pr_auc | avg_latency_ms | test_top1 | output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| videomae | 0.500 | 0.000 | 0.000 | 0.333 | 0.545 | 0.565 | 232.5 | 0.500 | `experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo` |
| slowfast | 0.375 | 0.273 | 0.150 | 0.342 | 0.253 | 0.382 | 383.2 | 0.375 | `experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo` |
| x3d | 0.500 | 0.500 | 1.000 | 0.333 | 0.450 | 0.516 | 271.1 | 0.500 | `experiments/accident_mmaction2_baseline/outputs/20260707_1212_x3d_inference_demo` |

Interpretation:

- VideoMAE smoke collapsed to non-accident: accident recall is 0.
- SlowFast produced mixed predictions but accident recall is low at 0.15.
- X3D predicted all clips as accident: accident recall is 1.0 but false alarm rate is unacceptable.
- These results validate the MMAction2 data/model/eval loop only; they are not deployment performance.

Recommendation:

- Next run must use pretrained checkpoints and real normal/hard-negative CCTV before drawing model-quality conclusions.
- Keep YOLO/Track as candidate/evidence inputs, not accident labels.
- Tune thresholds against validation PR curves and false alarms per hour.
