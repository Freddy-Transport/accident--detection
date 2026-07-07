# ACCIDENT 官方数据与 Demo 复现

本项目通过远程 AutoDL 复现 `accidentbench/ACCIDENT` 官方数据下载和 heuristic baseline demo。所有数据、模型、日志和输出均保存在远程服务器，不保存到本地。

## 路径

- 本项目：`/root/autodl-tmp/traffic_accident_rnd`
- 官方仓库：`/root/autodl-tmp/traffic_accident_rnd/third_party/ACCIDENT`
- 官方数据：`/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset`
- 项目软链接：`/root/autodl-tmp/traffic_accident_rnd/data/official_accident`
- 日志：`/root/autodl-tmp/traffic_accident_rnd/logs/accident_official`
- Demo 输出：`/root/autodl-tmp/traffic_accident_rnd/outputs/accident_official_demo`

## 运行顺序

```bash
cd /root/autodl-tmp/traffic_accident_rnd
scripts/accident_setup_official.sh
scripts/accident_download_dataset.sh
scripts/accident_run_demo.sh
```

Kaggle token 使用 `/root/.kaggle/access_token`，权限必须为 `600`。脚本不打印 token 内容。


## ONNX bbox demo fallback

`yolo11x.pt` 从 GitHub release 下载速度极慢时，可以使用已上传到远程数据盘的 `cardet.onnx`：

- 模型路径：`/root/autodl-tmp/traffic_accident_rnd/models/pretrained/cardet.onnx`
- SHA256：`9aa4b06f41c0de22c344ac00fb1a6e08f089097e63eb3c9cb88323f49984d97f`
- 运行依赖：heuristic `.venv` 额外安装 `lap`、`onnx`、`onnxruntime`
- ONNX 固定 batch=1，demo 默认 `ACCIDENT_BBOX_BATCH_SIZE=1`

```bash
cd /root/autodl-tmp/traffic_accident_rnd
ACCIDENT_BBOX_MODEL_PATH=/root/autodl-tmp/traffic_accident_rnd/models/pretrained/cardet.onnx ACCIDENT_BBOX_BATCH_SIZE=1 scripts/accident_run_demo.sh
```

说明：官方 `bbox_dynamics.py` 对 PyTorch `.pt` 权重调用 `YOLO.to()` 和 `track()`；导出的 ONNX 模型不支持该路径。本项目提供 `scripts/accident_bbox_dynamics_export_model.py`，只替换官方 `Tracker`，保留官方 temporal/spatial 评估流程。ONNX 路径使用 `predict()` 获取每帧 bbox，并过滤非法框。

## 验收点

- `scripts/env_check.sh` 和 `python -m pytest -q` 通过，确认现有 MVP 未破坏。
- `logs/accident_official/kaggle_files_picekl_accident.csv` 存在，说明 Kaggle 数据集可访问。
- `logs/accident_official/dataset_validation.json` 记录 `metadata-real.csv` 行数、视频数量和首个视频可打开状态。
- `outputs/accident_official_demo/demo_summary.json` 汇总第一个真实视频的 naive、optical flow 和 bbox dynamics 结果。
