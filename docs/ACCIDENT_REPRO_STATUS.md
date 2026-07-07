# ACCIDENT 复现状态

## 环境配置 - 2026-07-07

- 官方仓库路径：`/root/autodl-tmp/traffic_accident_rnd/third_party/ACCIDENT`
- 官方 upstream commit：`38b22973ec49449c18ccbf8db2328ce071eb77b0`
- 数据目标路径：`/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset`
- Kaggle token：已配置在远程 `/root/.kaggle/access_token`，权限 `600`，未写入 git。
- Heuristic 环境：`third_party/ACCIDENT/baselines/heuristic/.venv`
- Import 检查：`cv2 4.13.0;pandas 3.0.0;ruptures v1.1.10;torch 2.10.0+cu128 cuda True;ultralytics 8.4.11;`

## 已验证命令

```bash
scripts/accident_setup_official.sh
third_party/ACCIDENT/baselines/heuristic/.venv/bin/python naive.py --help
third_party/ACCIDENT/baselines/heuristic/.venv/bin/python optical_flow.py --help
third_party/ACCIDENT/baselines/heuristic/.venv/bin/python bbox_dynamics.py --help
```

## 日志

- `logs/accident_official/accident_upstream_commit.log`
- `logs/accident_official/uv_install_dataset_requirements.log`
- `logs/accident_official/uv_sync_heuristic.log`
- `logs/accident_official/heuristic_import_check.log`

## 数据下载与校验 - 2026-07-07

- Kaggle 数据集：`picekl/accident`
- 下载脚本：`scripts/accident_download_dataset.sh`
- 数据目录：`/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset`
- 项目软链接：`data/official_accident`
- 磁盘占用：`54G	/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset`
- 真实视频：`2027` 个
- metadata：`metadata-real.csv` 已规范为 baseline 兼容的文件名路径；原始文件保留为 `metadata-real.original.csv`
- 数据校验摘要：`{   "dataset_root": "/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset",   "metadata_real_exists": true,   "metadata_real_rows": 2027,   "real_video_count": 2027,   "first_video": "/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset/real_videos/Z4kg2Ev3vhk_00.mp4",   "first_video_opened": true,   "first_video_frame_count": 425,   "first_video_fps": 14.166666666666666,   "has_metadata_synthetic": true,   "has_synthetic_videos": true,   "metadata_original_backup": true } `
- optical flow 路径探针：`logs/accident_official/optical_flow_path_probe.log`，`--take 1` 成功。
