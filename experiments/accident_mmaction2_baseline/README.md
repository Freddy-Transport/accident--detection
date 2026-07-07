# ACCIDENT MMAction2 Baseline

This experiment builds clip-level video recognition baselines for ACCIDENT. Because ACCIDENT real is accident-centric, `non_accident` means weak pre-event clips, not explicit normal traffic videos.

Run order:

```bash
cd /root/autodl-tmp/traffic_accident_rnd
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/audit_dataset.py
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/build_mmaction_annotations.py --max-per-split 20
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/check_videos.py --per-class 20
experiments/accident_mmaction2_baseline/scripts/setup_mmaction_env.sh
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_sanity_check.py --model videomae --max-batches 10
```
## Current Smoke Results

| model | acc | accident_precision | accident_recall | macro_f1 | roc_auc | pr_auc | avg_latency_ms | test_top1 | output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| videomae | 0.500 | 0.000 | 0.000 | 0.333 | 0.545 | 0.565 | 232.5 | 0.500 | `experiments/accident_mmaction2_baseline/outputs/20260707_1208_videomae_inference_demo` |
| slowfast | 0.375 | 0.273 | 0.150 | 0.342 | 0.253 | 0.382 | 383.2 | 0.375 | `experiments/accident_mmaction2_baseline/outputs/20260707_1210_slowfast_inference_demo` |
| x3d | 0.500 | 0.500 | 1.000 | 0.333 | 0.450 | 0.516 | 271.1 | 0.500 | `experiments/accident_mmaction2_baseline/outputs/20260707_1212_x3d_inference_demo` |

Reports are under `experiments/accident_mmaction2_baseline/reports/`. Generated checkpoints, predictions and visualizations are under `experiments/accident_mmaction2_baseline/outputs/` and are intentionally git-ignored.
