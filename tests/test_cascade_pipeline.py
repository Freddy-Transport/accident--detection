from traffic_accident_rnd.cascade import (
    assign_iou_tracks,
    build_accident_evidence_tracks,
    build_final_accident_events,
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


def test_track_rules_emit_timestamped_speed_drop_without_annotations():
    speeds = [32.0, 31.0, 30.0, 29.0, 4.0, 3.0, 2.0]
    frames = [
        {
            "frame_index": idx,
            "timestamp_sec": idx * 0.5,
            "detections": [{
                "track_id": 11, "class_name": "car", "confidence": 0.9,
                "bbox_xyxy": [idx * 5, 0, idx * 5 + 20, 20],
                "speed_px_per_sec": speed,
            }],
        }
        for idx, speed in enumerate(speeds)
    ]

    segments = build_track_candidate_segments(frames, video_id="unknown")
    speed_events = [item for item in segments if "speed_drop" in item["trigger_reasons"]]

    assert speed_events
    assert speed_events[0]["trigger_time_sec"] == speed_events[0]["peak_time_sec"]
    assert speed_events[0]["evidence_track_ids"] == [11]
    assert "accident_time" not in str(speed_events[0])


def test_track_rules_emit_trajectory_conflict_with_both_track_ids():
    frames = []
    for idx in range(5):
        ax = idx * 10.0
        bx = 100.0 - idx * 10.0
        frames.append({
            "frame_index": idx,
            "timestamp_sec": float(idx),
            "detections": [
                {"track_id": 1, "class_name": "car", "bbox_xyxy": [ax, 0, ax + 10, 10],
                 "speed_px_per_sec": 10.0},
                {"track_id": 2, "class_name": "car", "bbox_xyxy": [bx, 0, bx + 10, 10],
                 "speed_px_per_sec": 10.0},
            ],
        })

    segments = build_track_candidate_segments(frames, video_id="unknown")
    conflicts = [item for item in segments if "trajectory_conflict" in item["trigger_reasons"]]

    assert conflicts
    assert conflicts[0]["evidence_track_ids"] == [1, 2]
    assert conflicts[0]["trigger_time_sec"] > 0.0



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



def test_build_final_accident_events_uses_score_threshold_not_pred_label():
    predictions = [
        {"candidate_id": 0, "segment_start_sec": 0.0, "segment_end_sec": 3.0, "accident_score": 0.55, "pred_label": 1, "trigger_reasons": ["bbox_overlap"], "evidence_track_ids": [1]},
        {"candidate_id": 1, "segment_start_sec": 4.0, "segment_end_sec": 7.0, "accident_score": 0.80, "pred_label": 1, "trigger_reasons": ["trajectory_conflict"], "evidence_track_ids": [2]},
    ]

    events = build_final_accident_events(predictions, video_id="demo", video_path="demo.mp4", threshold=0.56, checkpoint="best.pth")

    assert len(events) == 1
    assert events[0]["candidate_id"] == 1
    assert events[0]["event_type"] == "traffic_accident"
    assert events[0]["trigger_reasons"] == ["trajectory_conflict"]



def test_build_accident_evidence_tracks_labels_only_positive_evidence_tracks():
    predictions = [
        {
            "candidate_id": 0,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.0,
            "accident_score": 0.80,
            "trigger_reasons": ["abnormal_stop"],
            "evidence_track_ids": [7],
        },
        {
            "candidate_id": 1,
            "segment_start_sec": 2.0,
            "segment_end_sec": 3.0,
            "accident_score": 0.20,
            "trigger_reasons": ["speed_drop"],
            "evidence_track_ids": [9],
        },
    ]
    frames = [
        {
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "detections": [
                {"track_id": 7, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 1, 10, 11]},
                {"track_id": 8, "class_name": "car", "confidence": 0.8, "bbox_xyxy": [20, 1, 30, 11]},
            ],
        },
        {
            "frame_index": 20,
            "timestamp_sec": 2.0,
            "detections": [
                {"track_id": 9, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [40, 1, 50, 11]},
            ],
        },
    ]

    evidence = build_accident_evidence_tracks(predictions, frames, threshold=0.56)

    assert len(evidence) == 1
    assert evidence[0]["track_id"] == 7
    assert evidence[0]["frame_index"] == 0
    assert evidence[0]["label"] == "SUSPECT_ACCIDENT_VEHICLE"
    assert evidence[0]["accident_score"] == 0.80
    assert evidence[0]["trigger_reasons"] == ["abnormal_stop"]


def test_build_accident_evidence_tracks_does_not_label_without_evidence_track_ids():
    predictions = [
        {
            "candidate_id": 0,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.0,
            "accident_score": 0.90,
            "trigger_reasons": ["unknown"],
            "evidence_track_ids": [],
        }
    ]
    frames = [
        {
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "detections": [
                {"track_id": 1, "class_name": "car", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            ],
        }
    ]

    evidence = build_accident_evidence_tracks(predictions, frames, threshold=0.56)

    assert evidence == []

def test_final_events_deduplicate_overlapping_positive_windows() -> None:
    predictions = [
        {
            'candidate_id': 0, 'segment_start_sec': 0.0, 'segment_end_sec': 6.0,
            'trigger_time_sec': 2.0, 'candidate_score': 4.0, 'accident_score': 0.7,
            'trigger_reasons': ['speed_drop'], 'primary_evidence_track_ids': [1],
            'evidence_track_ids': [1],
        },
        {
            'candidate_id': 1, 'segment_start_sec': 4.0, 'segment_end_sec': 10.0,
            'trigger_time_sec': 6.0, 'candidate_score': 5.0, 'accident_score': 0.9,
            'trigger_reasons': ['trajectory_conflict'], 'primary_evidence_track_ids': [2, 3],
            'evidence_track_ids': [2, 3],
        },
    ]

    events = build_final_accident_events(
        predictions, video_id='demo', video_path='demo.mp4', threshold=0.56, checkpoint='model.pth')

    assert len(events) == 1
    assert events[0]['candidate_id'] == 1
    assert events[0]['detected_time_sec'] == 6.0
    assert events[0]['evidence_track_ids'] == [2, 3]
    assert events[0]['supporting_candidate_ids'] == [0, 1]
    assert events[0]['all_trigger_track_ids'] == [1, 2, 3]

