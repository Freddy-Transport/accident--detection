# YOLO/Track 输出 JSONL 规范

每行对应一个视频帧，保留检测和跟踪输出，供候选触发器和推理服务后续融合。

```json
{
  "frame_index": 12,
  "timestamp_sec": 0.4,
  "detections": [
    {
      "track_id": "car-7",
      "class_name": "car",
      "confidence": 0.91,
      "bbox_xyxy": [100.0, 80.0, 180.0, 140.0]
    }
  ]
}
```

## 约束

- `frame_index` 从 0 开始递增。
- `timestamp_sec` 单位为秒。
- `bbox_xyxy` 为 `[x1, y1, x2, y2]`，要求 `x2 > x1` 且 `y2 > y1`。
- `track_id` 可为 string、number 或 null；缺失跟踪 ID 时仍保留检测框。
