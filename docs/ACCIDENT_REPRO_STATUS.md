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
