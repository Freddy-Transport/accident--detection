# 视频数据 Manifest 规范

Manifest 使用 JSON Lines，每行描述一个视频样本或一个已裁剪片段。路径必须是远程服务器上的绝对路径或相对项目根目录的路径，不允许指向本地工作站路径。

## 必填字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `video_id` | string | 样本唯一 ID |
| `video_path` | string | 视频路径，推荐项目内相对路径 |
| `label` | string | `accident`、`normal` 或 `unknown` |
| `accident_start_sec` | number/null | 事故开始秒；非事故可为 null |
| `accident_end_sec` | number/null | 事故结束秒；非事故可为 null |
| `split` | string | `train`、`val`、`test` 或 `smoke` |
| `source_dataset` | string | 数据来源，如 `ACCIDENT`、`CADP`、`internal` |
| `camera_type` | string | 本项目默认 `fixed_traffic_surveillance` |
| `fps` | number/null | 视频帧率 |
| `duration_sec` | number/null | 视频时长秒 |
| `sha256` | string/null | 视频文件校验值 |
| `track_path` | string/null | 可选 YOLO/Track JSONL 路径 |

## 约束

- `accident_start_sec` 不得大于 `accident_end_sec`。
- `duration_sec` 存在时必须大于 0。
- `label=accident` 的样本应提供事故时间窗；公开数据缺失时间标注时先用 `unknown`，不要编造。
- 数据划分写入 manifest，不依赖隐式目录名。
