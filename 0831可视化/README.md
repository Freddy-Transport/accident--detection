# 0831 事故检测可视化 Demo

本目录只用于测试与汇报展示，不执行训练。模型使用全量 ACCIDENT 微调得到的
`best_acc_top1_epoch_2.pth`，事故判定阈值为 `0.56`。

## 展示结果

| 数据 | 类型 | VideoMAE 事故分数 | 判定 | 事故车辆框来源 |
| --- | --- | ---: | --- | --- |
| `-6SQSDj8cYU_00__accident.mp4` | 真实 CCTV，t-bone | 0.8416 | accident | YOLO + StrongSORT 轨迹与官方事故区域关联 |
| `Town06_sideswipe_clear_26.mp4` | CARLA 虚拟，sideswipe | 0.6173 | accident | CARLA 碰撞 actor 371、373 的逐帧二维框 |

两段视频均在每一帧显示红色事故车辆框。真实视频红框是与官方事故区域关联的疑似事故车辆
轨迹，不是框级事故真值；虚拟视频红框是 CARLA 碰撞 actor 真值框。VideoMAE 只输出片段级
事故概率，不直接输出车辆框。

## 主要文件

- `outputs/real/real_accident_demo_visualization.mp4`
- `outputs/synthetic/synthetic_accident_demo_visualization.mp4`
- `outputs/real/real_contact_sheet.jpg`
- `outputs/synthetic/synthetic_contact_sheet.jpg`
- `reports/0831事故检测可视化报告.md`
- `reports/demo_results.json`

## 复现

在远程项目根目录执行：

```bash
cd /root/autodl-tmp/traffic_accident_rnd
bash '0831可视化/scripts/run_0831_demo.sh'
```

脚本只执行推理、检测、跟踪和渲染，不包含训练命令。当前实例没有挂载 GPU 时会使用 CPU。
GPU 恢复后可设置：

```bash
DEMO_DEVICE=cuda:0 YOLO_DEVICE=0 bash '0831可视化/scripts/run_0831_demo.sh'
```

仅重新渲染已有结果：

```bash
SKIP_INFERENCE=1 bash '0831可视化/scripts/run_0831_demo.sh'
