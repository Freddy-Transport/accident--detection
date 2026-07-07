from traffic_accident_rnd.cascade import (
    assign_iou_tracks,
    build_legacy_candidate_segments,
    build_track_candidate_segments,
    iou_xyxy,
)


def test_iou_xyxy_returns_expected_overlap():
    assert round(iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 3) == 0.143
    assert iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_assign_iou_tracks_keeps_vehicle_id_across_frames():
    frames = [
        {
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "detections": [
                {"class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            ],
        },
        {
            "frame_index": 1,
            "timestamp_sec": 0.1,
            "detections": [
                {"class_name": "car", "confidence": 0.9, "bbox_xyxy": [1, 0, 11, 10]},
            ],
        },
    ]

    tracked = assign_iou_tracks(frames, iou_threshold=0.3)

    assert tracked[0]["detections"][0]["track_id"] == tracked[1]["detections"][0]["track_id"]
    assert tracked[1]["detections"][0]["speed_px_per_sec"] > 0


def test_build_track_candidate_segments_flags_abnormal_stop():
    frames = []
    for idx in range(6):
        frames.append({
            "frame_index": idx,
            "timestamp_sec": float(idx),
            "detections": [
                {
                    "track_id": 7,
                    "class_name": "car",
                    "confidence": 0.95,
                    "bbox_xyxy": [100, 100, 140, 140],
                    "speed_px_per_sec": 0.0 if idx >= 2 else 20.0,
                }
            ],
        })

    segments = build_track_candidate_segments(
        frames,
        video_id="demo",
        abnormal_stop_sec=3.0,
        pre_window_sec=1.0,
        post_window_sec=2.0,
    )

    assert segments
    assert segments[0]["send_to_video_model"] is True
    assert "abnormal_stop" in segments[0]["trigger_reasons"]
    assert 7 in segments[0]["evidence_track_ids"]



def test_build_legacy_candidate_segments_maps_track_ids_to_time_window():
    frames = [
        {"frame_index": 0, "timestamp_sec": 0.0, "detections": [{"track_id": 3, "bbox_xyxy": [0, 0, 10, 10]}]},
        {"frame_index": 5, "timestamp_sec": 1.0, "detections": [{"track_id": 3, "bbox_xyxy": [1, 0, 11, 10]}]},
    ]
    trajectory_events = {
        "events": [
            {
                "reason": "trajectory_conflict",
                "legacy_flag_index": 1,
                "legacy_event_type": "trajectory_anomaly_ids",
                "evidence_track_ids": [3],
                "candidate_score": 2.0,
            }
        ]
    }

    segments = build_legacy_candidate_segments(frames, trajectory_events, video_id="demo", pre_window_sec=0.5, post_window_sec=0.5)

    assert segments[0]["trigger_reasons"] == ["trajectory_conflict"]
    assert segments[0]["evidence_track_ids"] == [3]
    assert segments[0]["segment_start_sec"] == 0.0
    assert segments[0]["segment_end_sec"] == 1.5
