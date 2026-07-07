# VideoMAE Full ACCIDENT Baseline Report

## Data

- Full profile clips: train `625`, val `153`, test `2468`.
- Train labels: `{'non_accident': 219, 'accident': 406}`; val labels: `{'accident': 101, 'non_accident': 52}`; test labels: `{'non_accident': 948, 'accident': 1520}`.
- Group leakage count: `0`.
- `non_accident` is derived from pre-event windows; no real CCTV hard negative class is trained in this stage.

## Training

- Config: `/root/autodl-tmp/traffic_accident_rnd/experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py`.
- Work dir: `/root/autodl-tmp/traffic_accident_rnd/experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch`.
- Best checkpoint: `experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth`.
- Epochs: `3`; batch size: `1`; optimizer: AdamW; pretrained: Kinetics VideoMAE-B backbone.

## Validation Thresholds

- Default threshold 0.5 val accident recall `0.9505`, precision `0.7742`, ROC-AUC `0.8393`, PR-AUC `0.8958`.
- Val high-recall operating point: threshold `0.56`, accident recall `0.9109`, precision `0.8000`, FPR `0.4423`.
- Val macro-F1 operating point: threshold `0.67`, macro F1 `0.7763`.

## Test Metrics

| Threshold | Accuracy | Accident Precision | Accident Recall | Macro F1 | FPR | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.7382 | 0.7225 | 0.9336 | 0.6848 | 0.5749 | 0.7954 | 0.8452 |
| 0.56 val high recall | 0.7403 | 0.7448 | 0.8796 | 0.7056 | 0.4831 | 0.7954 | 0.8452 |
| 0.67 val macro F1 | 0.7196 | 0.8141 | 0.7059 | 0.7132 | 0.2584 | 0.7954 | 0.8452 |

- Test sweep best macro-F1 threshold `0.61` with macro F1 `0.7228`; this is reported for analysis, not selected as deployment threshold.
- Average test clip latency: `98.395 ms`; throughput: `10.163 clips/s`.
- Confusion matrix at threshold 0.56: `[[490, 458], [183, 1337]]`.

## Cascade Demo

- Negative/original-video demo score: `0.4742` at threshold `0.56`; positive windows `0`.
- Positive clip demo score: `0.8841` at threshold `0.56`; output video `experiments/accident_mmaction2_baseline/outputs/20260707_145438_cascade_positive_demo/uC7g8msxRak_00__accident_accident_visualization.mp4`.
- Positive trigger reasons: `['bbox_overlap', 'queue_growth', 'speed_drop']`.

## Interpretation

- Full fine-tuning improves over the earlier medium smoke baseline and gives usable accident recall, but false positives remain high because negative samples are weak pre-event windows rather than real hard negatives.
- For deployment, keep YOLO/Track as candidate/evidence only, and add real hard negative CCTV clips before using the model as an alarm source.
