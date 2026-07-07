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

## 候选片段 JSONL

候选触发器每行输出一个候选片段，只表示“送入视频模型复核”，不表示事故最终结论。

```json
{
  "video_id": "camera_demo",
  "segment_start_sec": 3.0,
  "segment_end_sec": 9.0,
  "peak_time_sec": 5.2,
  "candidate_score": 2.4,
  "trigger_reasons": ["speed_drop", "abnormal_stop"],
  "evidence_track_ids": [3, 7],
  "send_to_video_model": true,
  "evidence": {"method": "track_dynamics"}
}
```

当前候选触发器只作为 YOLO/Track 到 VideoMAE 的筛选接口；最终 accident / non_accident 由微调后的 VideoMAE 输出。
