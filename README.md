<p align="right">
  <a href="./README.md"><img src="https://img.shields.io/badge/语言-中文-0A7BBC?style=for-the-badge" alt="中文"></a>
  <a href="./README_EN.md"><img src="https://img.shields.io/badge/Language-English-555555?style=for-the-badge" alt="English"></a>
</p>

<h1 align="center">城市道路监控交通事故早期发现</h1>

<p align="center">
  面向固定道路 CCTV 的候选片段触发、轨迹证据分析与 VideoMAE 事故语义识别流程
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MMAction2-1.2.0-1F6FEB" alt="MMAction2 1.2.0">
  <img src="https://img.shields.io/badge/VideoMAE-ACCIDENT_Fine--tuned-C43E1C" alt="VideoMAE fine-tuned">
  <img src="https://img.shields.io/badge/Tracking-StrongSORT%20%7C%20IoU-2E8B57" alt="StrongSORT or IoU">
</p>

<p align="center">
  <img src="./docs/assets/accident_pipeline_demo.jpg" width="760" alt="事故检测级联可视化示例">
</p>

<p align="center">
  <sub>红框表示触发候选片段的疑似事故证据车辆，不是框级事故真值；最终事故结论由 VideoMAE 分数决定。</sub>
</p>

## 项目简介

本项目面向城市道路固定监控视频中的交通事故早期发现，不面向自动驾驶行车记录仪事故预判。系统先用 YOLO、轨迹跟踪和异常规则筛选疑似片段，再由基于 MMAction2 微调的 VideoMAE 完成事故语义判断，最后输出事件 JSON、证据轨迹和标注视频。

当前仓库已形成从数据审计、MMAction2 标注、VideoMAE 微调评估，到新视频端到端推理和 TorchScript/ONNX 导出的可复现闭环。

## 核心能力

- 支持本地视频路径以及 RTSP/RTMP/HTTP 视频流输入。
- YOLO 检测车辆、行人和非机动车，并输出逐帧 JSONL。
- StrongSORT 跟踪车辆；依赖不可用时可自动回退到轻量 IoU Tracker。
- 复用轨迹异常逻辑，生成急减速、异常停车、轨迹冲突和队列增长等候选原因。
- 使用 ACCIDENT 数据集微调的 VideoMAE-B 对候选片段做二分类。
- 输出最终事故事件、模型分数、证据车辆、完整中间结果和可视化视频。
- 支持 VideoMAE TorchScript 与 ONNX 导出，并完成 logits 数值一致性验证。
- 事件推送默认使用 dry-run，只有显式配置后才会访问业务接口。

## 系统架构

```mermaid
flowchart LR
    A[视频文件或视频流] --> B[YOLO 目标检测]
    B --> C[StrongSORT / IoU 跟踪]
    C --> D[轨迹异常判断]
    D --> E{生成候选片段?}
    E -- 否 --> F[记录证据并结束]
    E -- 是 --> G[VideoMAE 事故识别]
    G --> H{accident_score >= 阈值?}
    H -- 否 --> I[保存非事故预测]
    H -- 是 --> J[事故事件输出]
    C --> K[车辆框与 Track ID]
    K --> J
    J --> L[final_events.json]
    J --> M[visualization.mp4]
    J --> N[可选事件推送]
```

> YOLO、Track 和轨迹规则只负责候选触发与结构化证据，不会被包装成事故识别结论。只有候选片段的 VideoMAE 分数达到阈值，系统才输出事故事件。

## 快速开始

### 1. 准备环境

参考环境为 Python 3.12、PyTorch 2.8.0+cu128、MMCV 2.1.0、MMEngine 0.10.7、MMAction2 1.2.0 和 decord 0.6.0。

```bash
cd /root/autodl-tmp/traffic_accident_rnd
hostname
pwd
nvidia-smi || true

python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 创建隔离的 MMAction2 环境
bash experiments/accident_mmaction2_baseline/scripts/setup_mmaction_env.sh

# 准备 ACCIDENT heuristic / Ultralytics 环境
bash scripts/accident_setup_official.sh
```

环境检查会把主机、磁盘、Python、PyTorch、GPU 和 Git 版本写入日志：

```bash
bash scripts/env_check.sh
```

### 2. 准备模型文件

模型权重和数据集不提交到 Git。默认推理入口需要以下远程文件：

| 用途 | 默认路径 |
| --- | --- |
| 车辆检测权重 | `models/pretrained/车辆检测_v8l.pt` |
| VideoMAE 最佳 checkpoint | `experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth` |
| TorchScript 导出模型 | `models/exported/mmaction2/videomae_full/videomae_full.ts.pt` |
| ONNX 导出模型 | `models/exported/mmaction2/videomae_full/videomae_full.onnx` |

如模型位于其他目录，可直接使用 `run_event_pipeline.py` 的 `--yolo-model`、`--checkpoint` 和 `--config` 参数覆盖默认值。

### 3. 上传并检测新视频

新视频应放在远程数据盘：

```text
/autodl-fs/data/traffic_accident_rnd/user_videos/
```

运行单个视频：

```bash
export OMP_NUM_THREADS=1

.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video /autodl-fs/data/traffic_accident_rnd/user_videos/demo.mp4 \
  --threshold 0.56 \
  --tracker auto \
  --frame-stride 5 \
  --device cuda:0
```

检测上传目录中最新的视频：

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --latest
```

视频流输入：

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py \
  --video "rtsp://user:password@camera/stream" \
  --tracker auto \
  --device cuda:0
```

完整参数：

```bash
.venv/bin/python experiments/accident_mmaction2_baseline/scripts/run_user_video.py --help
```

## 推理输出

每次运行会创建独立目录：

```text
experiments/accident_mmaction2_baseline/outputs/YYYYMMDD_HHMM_event_pipeline_<video_id>/
```

| 文件 | 内容 |
| --- | --- |
| `final_events.json` | 最终事故事件，仅包含达到 VideoMAE 阈值的候选片段 |
| `visualization.mp4` | 事故横幅、疑似证据车辆红框、Track ID 和模型分数 |
| `accident_evidence_tracks.json` | 可视化使用的逐帧证据车辆 |
| `videomae_predictions.json` | 候选片段的 accident score 和分类结果 |
| `candidate_segments.jsonl` | 候选时间段、触发原因和证据 Track ID |
| `trajectory_events.json` | 轨迹异常事件 |
| `detections.jsonl` / `tracks.jsonl` | YOLO 检测和跟踪中间结果 |
| `pipeline_config.json` | 本次运行参数快照 |
| `logs/` | 每个流水线步骤的命令、耗时、返回码和输出 |

事件推送默认关闭。启用真实推送时必须同时提供 `--push`、`TRAFFIC_EVENT_PUSH_ENABLED=1` 和推送地址；否则只生成 `event_push_dry_run.json`。

## 数据与模型

当前主实验使用 [ACCIDENT](https://github.com/accidentbench/ACCIDENT) 固定交通监控数据集，并按事件时间派生 clip-level 二分类样本：

- `0 non_accident`：事故前安全窗口，属于弱负样本。
- `1 accident`：事故事件窗口。
- 同一原始视频不会跨 train/val/test，已检查 group leakage 为 0。
- 当前没有真实 CCTV hard negative 类别，不汇报 hard negative 指标。

全量 profile：

| Split | Clips | 说明 |
| --- | ---: | --- |
| Train | 625 | 219 non-accident / 406 accident |
| Validation | 153 | 52 non-accident / 101 accident |
| Test | 2468 | 948 non-accident / 1520 accident |

MMAction2 配置位于 `experiments/accident_mmaction2_baseline/configs/`，包含 VideoMAE、SlowFast 和 X3D；当前端到端入口默认使用全量微调后的 VideoMAE-B。

## Baseline 结果

VideoMAE-B 使用 Kinetics 预训练权重，在 ACCIDENT full profile 上微调 3 个 epoch。下表来自 test split，部署默认采用验证集选择的高召回阈值 `0.56`。

| Threshold | Accuracy | Accident Precision | Accident Recall | Macro F1 | FPR | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.7382 | 0.7225 | 0.9336 | 0.6848 | 0.5749 | 0.7954 | 0.8452 |
| **0.56** | **0.7403** | **0.7448** | **0.8796** | **0.7056** | **0.4831** | **0.7954** | **0.8452** |
| 0.67 | 0.7196 | 0.8141 | 0.7059 | 0.7132 | 0.2584 | 0.7954 | 0.8452 |

测试集平均单 clip 延迟为 98.395 ms，吞吐为 10.163 clips/s。由于负样本主要是事故前窗口，这些结果不能直接代表真实道路上线后的误报率。

详细报告：[VideoMAE full baseline](./experiments/accident_mmaction2_baseline/reports/videomae_full_baseline_report.md) · [三模型对比](./experiments/accident_mmaction2_baseline/reports/model_comparison_report.md) · [级联验证](./experiments/accident_mmaction2_baseline/reports/event_pipeline_fusion_report.md)

## 模型导出

```bash
.venv_mmaction/bin/python \
  experiments/accident_mmaction2_baseline/scripts/export_videomae_model.py \
  --config experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py \
  --checkpoint experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth \
  --device cuda:0 \
  --opset 17
```

导出输入为预处理后的 `float32[1, 3, 16, 224, 224]` RGB tensor，输出为 `float32[1, 2]` logits。已验证 TorchScript 最大绝对误差为 `0.0`，ONNX 为 `7.15e-7`。导出模型仍必须使用与 MMAction2 配置一致的采样、颜色空间和归一化流程。

## 项目结构

```text
traffic_accident_rnd/
├── configs/                         # MVP 触发器、模型和服务配置
├── data/                            # manifest 与远程数据盘软链接
├── docs/                            # 数据规范、运行手册和接口文档
├── experiments/accident_mmaction2_baseline/
│   ├── configs/                     # VideoMAE / SlowFast / X3D 配置
│   ├── data/annotations/            # MMAction2 annotation 与 manifest
│   ├── scripts/                     # 审计、训练、评估、级联和导出脚本
│   ├── reports/                     # 实验指标与复现报告
│   └── outputs/                     # checkpoint、预测和视频，Git 忽略
├── models/                          # 预训练与导出权重，Git 忽略
├── scripts/                         # 环境、数据集和基础 MVP 入口
├── src/traffic_accident_rnd/        # 触发器、模型、API 与级联公共逻辑
├── tests/                           # 单元和回归测试
└── third_party/highway_inference_legacy/
                                      # StrongSORT 与轨迹异常适配源码
```

## 测试

```bash
.venv/bin/python -m pytest -q
```

当前回归结果为 25 passed。环境和模型文件依赖远程路径，提交前应同时确认 `git status` 没有意外包含 checkpoint、视频或实验输出。

## 已知限制

- `non_accident` 是事故前弱负样本，不等价于真实 normal / hard negative CCTV。
- 红框是候选证据车辆，不是事故车辆框级标注或碰撞参与者真值。
- 当前视频流入口仍按单段任务运行，尚未实现持续缓冲、断流重连和多路摄像头调度。
- SlowFast 预训练权重与本地 lateral temporal kernel 存在部分不匹配，不能作为严格公平对比。
- 实际部署前仍需补充拥堵、临停、公交停靠、施工、夜间反光和雨雾等 hard negative 数据。
- 仓库不包含数据集、YOLO 权重、VideoMAE checkpoint、导出模型和生成视频。

## 文档

- [新视频运行手册](./experiments/accident_mmaction2_baseline/reports/user_video_runbook.md)
- [数据集审计报告](./experiments/accident_mmaction2_baseline/reports/dataset_audit_report.md)
- [MMAction2 环境报告](./experiments/accident_mmaction2_baseline/reports/mmaction2_setup_report.md)
- [模型导出报告](./experiments/accident_mmaction2_baseline/reports/model_export_report.md)
- [数据与 Track 接口规范](./docs/DATA_SPEC.md)
- [ACCIDENT 官方复现说明](./docs/ACCIDENT_REPRO.md)

## 致谢

本项目的数据和视频理解基线基于 [ACCIDENT](https://github.com/accidentbench/ACCIDENT) 与 [MMAction2](https://github.com/open-mmlab/mmaction2)，目标检测使用 [Ultralytics](https://github.com/ultralytics/ultralytics)。跟踪接口设计参考 [BoxMOT](https://github.com/mikel-brostrom/boxmot) 的可插拔跟踪工程实践。README 的组织方式也参考了这些高关注度开源项目的快速开始、能力边界、基准结果和模块化文档结构。
