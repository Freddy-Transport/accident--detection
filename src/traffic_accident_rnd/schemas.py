"""Schema validators for dataset manifests and YOLO/Track JSONL output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_MANIFEST_FIELDS = {
    "video_id",
    "video_path",
    "label",
    "accident_start_sec",
    "accident_end_sec",
    "split",
    "source_dataset",
    "camera_type",
    "fps",
    "duration_sec",
    "sha256",
    "track_path",
}

VALID_LABELS = {"accident", "normal", "unknown"}
VALID_SPLITS = {"train", "val", "test", "smoke"}
TRACK_FRAME_FIELDS = {"frame_index", "timestamp_sec", "detections"}


def _is_number_or_none(value: Any) -> bool:
    return value is None or isinstance(value, (int, float))


def validate_manifest_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for one manifest JSON object."""
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(record))
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")

    if "video_id" in record and not isinstance(record["video_id"], str):
        errors.append("video_id must be a string")
    if "video_path" in record and not isinstance(record["video_path"], str):
        errors.append("video_path must be a string")
    if record.get("label") not in VALID_LABELS:
        errors.append(f"label must be one of {sorted(VALID_LABELS)}")
    if record.get("split") not in VALID_SPLITS:
        errors.append(f"split must be one of {sorted(VALID_SPLITS)}")
    if "camera_type" in record and not isinstance(record["camera_type"], str):
        errors.append("camera_type must be a string")
    if "source_dataset" in record and not isinstance(record["source_dataset"], str):
        errors.append("source_dataset must be a string")

    start = record.get("accident_start_sec")
    end = record.get("accident_end_sec")
    if not _is_number_or_none(start):
        errors.append("accident_start_sec must be a number or null")
    if not _is_number_or_none(end):
        errors.append("accident_end_sec must be a number or null")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start > end:
        errors.append("accident_start_sec must be <= accident_end_sec")
    if record.get("label") == "accident" and (start is None or end is None):
        errors.append("accident samples should include accident_start_sec and accident_end_sec")

    fps = record.get("fps")
    duration = record.get("duration_sec")
    if not _is_number_or_none(fps) or (isinstance(fps, (int, float)) and fps <= 0):
        errors.append("fps must be a positive number or null")
    if not _is_number_or_none(duration) or (isinstance(duration, (int, float)) and duration <= 0):
        errors.append("duration_sec must be a positive number or null")
    if "sha256" in record and record["sha256"] is not None and not isinstance(record["sha256"], str):
        errors.append("sha256 must be a string or null")
    if "track_path" in record and record["track_path"] is not None and not isinstance(record["track_path"], str):
        errors.append("track_path must be a string or null")
    return errors


def validate_track_frame_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for one track frame JSON object."""
    errors: list[str] = []
    missing = sorted(TRACK_FRAME_FIELDS - set(record))
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")
    if "frame_index" in record and (not isinstance(record["frame_index"], int) or record["frame_index"] < 0):
        errors.append("frame_index must be a non-negative integer")
    timestamp = record.get("timestamp_sec")
    if "timestamp_sec" in record and (not isinstance(timestamp, (int, float)) or timestamp < 0):
        errors.append("timestamp_sec must be a non-negative number")
    detections = record.get("detections")
    if "detections" in record and not isinstance(detections, list):
        errors.append("detections must be a list")
        return errors
    for idx, det in enumerate(detections or []):
        if not isinstance(det, dict):
            errors.append(f"detections[{idx}] must be an object")
            continue
        if not isinstance(det.get("class_name"), str):
            errors.append(f"detections[{idx}].class_name must be a string")
        confidence = det.get("confidence")
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            errors.append(f"detections[{idx}].confidence must be between 0 and 1")
        bbox = det.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            errors.append(f"detections[{idx}].bbox_xyxy must contain four numbers")
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            errors.append(f"detections[{idx}].bbox_xyxy must satisfy x2 > x1 and y2 > y1")
    return errors


def _validate_jsonl_file(path: str | Path, validator) -> list[str]:
    path = Path(path)
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                errors.append(f"{path}:{line_no}: line must be a JSON object")
                continue
            for err in validator(record):
                errors.append(f"{path}:{line_no}: {err}")
    return errors


def validate_manifest_file(path: str | Path) -> list[str]:
    return _validate_jsonl_file(path, validate_manifest_record)


def validate_track_file(path: str | Path) -> list[str]:
    return _validate_jsonl_file(path, validate_track_frame_record)
