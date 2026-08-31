# 0831 事故检测可视化 Demo

本目录只用于测试与汇报展示，不执行训练。模型使用全量 ACCIDENT 微调得到的
`best_acc_top1_epoch_2.pth`，事故判定阈值为 `0.56`。

## 正确推理协议

```text
完整未知视频 -> YOLO -> StrongSORT
-> 速度突降/异常停止/轨迹冲突/框重叠/队列增长规则
-> 候选片段 -> VideoMAE -> 触发轨迹事故车辆标注
```

推理命令不接受 `accident_time`、事故区域或碰撞 actor 标注。只有在
`final_events.json` 写入完成后，独立评价脚本才读取这些标签，计算时刻误差和空间匹配。

默认入口：`scripts/run_unknown_full_video.sh`。批量测试入口：
`scripts/run_unknown_benchmark.sh`。结果写入 `unknown_runs/`。


## 本次无标注完整视频结果

结果目录：`unknown_runs/0831_corrected_unknown_full_video/`。

| 数据 | 轨迹候选 | 最终事件 | 自动发现时刻 | 事后真值 | 时刻误差 | 疑似轨迹 | 事后空间评价 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 真实 CCTV | 4 | 1 | 4.473 s | 5.083 s | 0.610 s 提前 | 3、8 | 命中，轨迹 8 IoU=0.785 |
| CARLA 虚拟 | 2 | 1 | 1.950 s | 5.550 s | 3.600 s 提前 | 2、3 | 未命中 |

真实样本完成了事故时刻发现和疑似事故车辆定位。虚拟样本仅完成事故语义检出，
YOLO/StrongSORT 未在官方事故区域提供可用触发轨迹，因此不能视为事故车辆定位成功。
这两个视频用于开发和展示，不是独立未参与调试的统计测试集。

可视化视频：

- `unknown_runs/0831_corrected_unknown_full_video/real/visualization.mp4`
- `unknown_runs/0831_corrected_unknown_full_video/synthetic/visualization.mp4`

详细报告：`reports/未知完整视频自动发现报告.md`。

## 历史标签辅助结果（不能证明自动事故发现）

| 数据 | 类型 | VideoMAE 事故分数 | 判定 | 事故车辆框来源 |
| --- | --- | ---: | --- | --- |
| `-6SQSDj8cYU_00__accident.mp4` | 真实 CCTV，t-bone | 0.8416 | accident | YOLO + StrongSORT 轨迹与官方事故区域关联 |
| `Town06_sideswipe_clear_26.mp4` | CARLA 虚拟，sideswipe | 0.6173 | accident | CARLA 碰撞 actor 371、373 的逐帧二维框 |

上述旧结果在候选生成或车辆关联时使用了事故标注，只能作为渲染样例，不能作为未知视频自动发现结果。
它们保留用于结果溯源，但默认运行入口已停止执行该流程。有效结果必须位于 `unknown_runs/`。
`pipeline_config.json` 和 `final_events.json` 必须声明 `annotation_free_inference=true`、`annotation_inputs=[]`。

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

单个完整未知视频：

```bash
bash '0831可视化/scripts/run_unknown_full_video.sh' /path/to/full_video.mp4
```

真实与虚拟完整视频批量测试并在推理后评价：

```bash
bash '0831可视化/scripts/run_unknown_benchmark.sh'
```

GPU 恢复后可显式设置：

```bash
MMACTION_DEVICE=cuda:0 YOLO_DEVICE=0 bash '0831可视化/scripts/run_unknown_full_video.sh' /path/to/full_video.mp4
```
