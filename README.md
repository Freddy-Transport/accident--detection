# 城市道路监控交通事故早期发现 MVP

本项目面向城市道路固定监控视频中的交通事故早期发现，不面向自动驾驶行车记录仪事故预判。所有代码、数据、模型、日志和文档均位于远程 AutoDL 数据盘项目目录：`/root/autodl-tmp/traffic_accident_rnd`。

## MVP 范围

- 工程目录与远程可复现环境记录
- 固定监控视频数据 manifest 规范
- YOLO/Track 输出接口预留
- 基于帧差 z-score 的事故候选片段触发器
- 基于 `torchvision.models.video.r3d_18` 的二分类视频 baseline
- FastAPI 推理服务 demo

## 远程运行原则

每次远程操作前执行并记录：

```bash
hostname
pwd
nvidia-smi || true
```

当前容器内 `nvidia-smi` 可能因 `/usr/bin/nvidia-smi` 不可执行而失败；该失败会写入日志，但 CPU smoke test 不因此中断。
