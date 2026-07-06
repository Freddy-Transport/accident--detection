from pathlib import Path

from traffic_accident_rnd.schemas import (
    REQUIRED_MANIFEST_FIELDS,
    validate_manifest_file,
    validate_manifest_record,
    validate_track_file,
    validate_track_frame_record,
)


def valid_manifest_record():
    return {
        "video_id": "cam01-0001",
        "video_path": "data/raw/cam01.mp4",
        "label": "accident",
        "accident_start_sec": 2.0,
        "accident_end_sec": 4.0,
        "split": "train",
        "source_dataset": "ACCIDENT",
        "camera_type": "fixed_traffic_surveillance",
        "fps": 25.0,
        "duration_sec": 8.0,
        "sha256": None,
        "track_path": None,
    }


def test_manifest_schema_lists_all_required_fields():
    assert "video_id" in REQUIRED_MANIFEST_FIELDS
    assert "track_path" in REQUIRED_MANIFEST_FIELDS
    assert len(REQUIRED_MANIFEST_FIELDS) == 12


def test_valid_manifest_record_has_no_errors():
    assert validate_manifest_record(valid_manifest_record()) == []


def test_manifest_record_rejects_missing_and_bad_time_window():
    record = valid_manifest_record()
    record.pop("video_path")
    record["accident_start_sec"] = 5.0
    record["accident_end_sec"] = 4.0
    errors = validate_manifest_record(record)
    assert any("video_path" in error for error in errors)
    assert any("accident_start_sec" in error for error in errors)


def test_valid_track_frame_has_no_errors():
    record = {
        "frame_index": 3,
        "timestamp_sec": 0.12,
        "detections": [
            {
                "track_id": "car-1",
                "class_name": "car",
                "confidence": 0.9,
                "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
            }
        ],
    }
    assert validate_track_frame_record(record) == []


def test_track_frame_rejects_invalid_bbox():
    record = {
        "frame_index": 3,
        "timestamp_sec": 0.12,
        "detections": [
            {
                "track_id": "car-1",
                "class_name": "car",
                "confidence": 0.9,
                "bbox_xyxy": [40.0, 20.0, 10.0, 60.0],
            }
        ],
    }
    errors = validate_track_frame_record(record)
    assert any("bbox_xyxy" in error for error in errors)


def test_manifest_and_track_files_validate(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"video_id":"v1","video_path":"data/raw/v1.mp4","label":"normal",'
        '"accident_start_sec":null,"accident_end_sec":null,"split":"test",'
        '"source_dataset":"internal","camera_type":"fixed_traffic_surveillance",'
        '"fps":25.0,"duration_sec":3.0,"sha256":null,"track_path":null}\n',
        encoding="utf-8",
    )
    track_path = tmp_path / "tracks.jsonl"
    track_path.write_text(
        '{"frame_index":0,"timestamp_sec":0.0,"detections":[]}\n',
        encoding="utf-8",
    )
    assert validate_manifest_file(manifest_path) == []
    assert validate_track_file(track_path) == []
