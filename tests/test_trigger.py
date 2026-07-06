import json
from pathlib import Path

from traffic_accident_rnd.trigger import detect_segments_from_scores, write_candidate_segments


def test_detect_segments_from_scores_groups_adjacent_motion_peaks():
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    scores = [0.1, 0.2, 9.0, 10.0, 0.3, 0.2]
    segments = detect_segments_from_scores(
        times,
        scores,
        threshold_z=1.0,
        pre_window_sec=0.5,
        post_window_sec=1.0,
        min_gap_sec=1.1,
        max_segments=3,
    )
    assert len(segments) == 1
    segment = segments[0]
    assert segment["segment_start_sec"] == 1.5
    assert segment["segment_end_sec"] == 4.0
    assert segment["peak_time_sec"] == 3.0
    assert segment["trigger_score"] > 1.0
    assert segment["evidence"]["peak_count"] == 2


def test_detect_segments_from_scores_returns_empty_without_peaks():
    segments = detect_segments_from_scores(
        [0.0, 1.0, 2.0],
        [1.0, 1.1, 0.9],
        threshold_z=5.0,
    )
    assert segments == []


def test_write_candidate_segments_outputs_jsonl(tmp_path: Path):
    path = tmp_path / "candidates.jsonl"
    write_candidate_segments(
        [
            {
                "video_id": "v1",
                "segment_start_sec": 1.0,
                "segment_end_sec": 2.0,
                "peak_time_sec": 1.5,
                "trigger_score": 3.2,
                "evidence": {"method": "frame_diff_zscore"},
            }
        ],
        path,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["video_id"] == "v1"
    assert rows[0]["evidence"]["method"] == "frame_diff_zscore"
