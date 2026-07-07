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

## 验收点

- `scripts/env_check.sh` 和 `python -m pytest -q` 通过，确认现有 MVP 未破坏。
- `logs/accident_official/kaggle_files_picekl_accident.csv` 存在，说明 Kaggle 数据集可访问。
- `logs/accident_official/dataset_validation.json` 记录 `metadata-real.csv` 行数、视频数量和首个视频可打开状态。
- `outputs/accident_official_demo/demo_summary.json` 汇总第一个真实视频的 naive、optical flow 和 bbox dynamics 结果。
