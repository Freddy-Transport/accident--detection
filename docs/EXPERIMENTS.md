# 实验记录

## 记录要求

每次实验记录以下信息：

- git commit hash
- 执行命令
- 配置文件路径
- 数据 manifest 路径和 split
- 模型权重路径
- 日志路径
- 指标或 smoke test 结果

## MVP 验证记录 - 2026-07-06

- 代码提交：`1522cfa`
- 项目目录：`/root/autodl-tmp/traffic_accident_rnd`
- 数据规范：`data/manifests/example_manifest.jsonl`，split 为 `smoke`
- Track 规范：`data/manifests/example_tracks.jsonl`
- 配置文件：`configs/paths.yaml`、`configs/trigger.yaml`、`configs/model.yaml`、`configs/service.yaml`、`configs/dataset.yaml`
- 环境日志：`logs/experiments/final_env_check_stdout.log`
- 单测日志：`logs/tests/final_pytest.log`
- Kaggle 检查：`logs/dataset/final_kaggle_check_stdout.log`，当前无 `/root/.kaggle/kaggle.json`，未下载公开数据
- Smoke 视频：`data/samples/smoke_accident_like.mp4`
- 候选片段：`outputs/candidates/final_smoke_candidates.jsonl`
- Smoke 模型权重：`models/checkpoints/final_smoke_r3d18.pt`
- CLI 推理输出：`outputs/inference/final_smoke_prediction.json`
- API 健康检查：`outputs/inference/api_health.json`
- API 推理输出：`outputs/inference/api_smoke_prediction.json`
- API 服务日志：`logs/service/api_smoke.log`

### 执行命令

```bash
scripts/env_check.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_manifest.py data/manifests/example_manifest.jsonl
.venv/bin/python scripts/validate_manifest.py --kind track data/manifests/example_tracks.jsonl
.venv/bin/python scripts/check_kaggle_dataset.py --dataset accidentbench/accident-benchmark --max-gb 5
.venv/bin/python scripts/generate_smoke_video.py --output data/samples/smoke_accident_like.mp4
.venv/bin/python scripts/run_trigger.py --video data/samples/smoke_accident_like.mp4 --output outputs/candidates/final_smoke_candidates.jsonl --threshold-z 1.5
.venv/bin/python scripts/train_baseline.py --smoke --output models/checkpoints/final_smoke_r3d18.pt
.venv/bin/python scripts/infer_video.py --video data/samples/smoke_accident_like.mp4 --checkpoint models/checkpoints/final_smoke_r3d18.pt --output outputs/inference/final_smoke_prediction.json --threshold-z 1.5
```

### 当前结果

- 远程 GPU：`nvidia-smi` 可用，设备为 RTX 4080 32GB。
- 单测：`tests/test_schemas.py`、`tests/test_trigger.py`、`tests/test_model_inference.py`、`tests/test_api.py` 全部通过。
- 数据下载：Kaggle token 缺失，按规则记录阻塞状态，未下载公开数据。
- 候选触发：smoke 视频生成至少 1 个事故候选片段。
- 模型 baseline：R3D-18 smoke 训练完成，权重保存到远程数据盘。
- 推理 demo：CLI 和 FastAPI 均返回至少 1 个候选片段及 `accident_score`。

### 下一步建议

1. 在远程配置 `/root/.kaggle/kaggle.json` 后重新运行 Kaggle 数据检查，确认数据集文件大小不超过 5GB 再下载。
2. 接入真实固定监控视频和 YOLO/Track JSONL，扩展 manifest 的 train/val/test split。
3. 用真实标注训练 R3D-18 分类头，并记录每次训练的 commit、manifest、配置、权重和日志。
