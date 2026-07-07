# VideoMAE Export Report

- Timestamp: `2026-07-07T16:13:59+0800`
- Config: `/root/autodl-tmp/traffic_accident_rnd/experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py`
- Checkpoint: `/root/autodl-tmp/traffic_accident_rnd/experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth`
- Input shape: `[1, 3, 16, 224, 224]`
- TorchScript: `ok` `/root/autodl-tmp/traffic_accident_rnd/models/exported/mmaction2/videomae_full/videomae_full.ts.pt` diff `0.0`
- ONNX: `ok` `/root/autodl-tmp/traffic_accident_rnd/models/exported/mmaction2/videomae_full/videomae_full.onnx` diff `7.152557373046875e-07`

## Notes

- Exported logits expect preprocessed RGB tensors with MMAction2 mean/std normalization.
- Keep MMAction2 config and preprocessing with the exported model; `.pt`/`.onnx` alone is not the full video pipeline.
