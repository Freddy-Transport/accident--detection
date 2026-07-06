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

## 当前环境基线

- 项目目录：`/root/autodl-tmp/traffic_accident_rnd`
- Python：项目 `.venv`，继承 `/root/miniconda3` 的 PyTorch/TorchVision
- GPU 状态：以 `scripts/env_check.sh` 日志为准；当前 `nvidia-smi` 失败时只执行 CPU smoke test
