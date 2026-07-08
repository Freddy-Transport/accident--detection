"""Utilities for the YOLO/Track -> candidate -> VideoMAE cascade."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "van"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iou_xyxy(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def bbox_center_xyxy(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _same_track_class(track_class: str | None, det_class: str | None) -> bool:
    if not track_class or not det_class:
        return True
    return track_class == det_class


def assign_iou_tracks(
    frames: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.3,
    max_age_frames: int = 8,
    vehicle_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assign stable ids to per-frame detections with a deterministic IoU tracker.

    This is a lightweight fallback for reproducible demos. It is not meant to
    replace StrongSORT/ByteTrack in production, but it preserves the JSONL
    contract needed by the accident candidate trigger.
    """
    allowed = vehicle_classes or VEHICLE_CLASSES
    active: dict[int, dict[str, Any]] = {}
    next_id = 1
    tracked_frames: list[dict[str, Any]] = []

    for frame in sorted(frames, key=lambda item: int(item.get("frame_index", 0))):
        frame_index = int(frame.get("frame_index", 0))
        timestamp = float(frame.get("timestamp_sec", 0.0))
        detections = [dict(det) for det in frame.get("detections", [])]
        detections.sort(key=lambda det: float(det.get("confidence", 0.0)), reverse=True)
        assigned_tracks: set[int] = set()
        output_dets: list[dict[str, Any]] = []

        for det in detections:
            class_name = str(det.get("class_name", ""))
            if allowed and class_name not in allowed:
                continue
            bbox = [float(v) for v in det.get("bbox_xyxy", [])]
            if len(bbox) != 4:
                continue
            best_id: int | None = None
            best_iou = 0.0
            for track_id, state in active.items():
                if track_id in assigned_tracks:
                    continue
                if frame_index - int(state["last_frame_index"]) > max_age_frames:
                    continue
                if not _same_track_class(str(state.get("class_name", "")), class_name):
                    continue
                score = iou_xyxy(bbox, state["bbox_xyxy"])
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None or best_iou < iou_threshold:
                best_id = next_id
                next_id += 1
                prev_center = bbox_center_xyxy(bbox)
                prev_time = timestamp
            else:
                prev_center = active[best_id]["center_xy"]
                prev_time = float(active[best_id]["timestamp_sec"])

            cx, cy = bbox_center_xyxy(bbox)
            dt = max(1e-6, timestamp - prev_time)
            speed = math.hypot(cx - float(prev_center[0]), cy - float(prev_center[1])) / dt if timestamp > prev_time else 0.0
            enriched = {
                **det,
                "track_id": int(best_id),
                "bbox_xyxy": bbox,
                "center_xy": [round(cx, 3), round(cy, 3)],
                "speed_px_per_sec": round(float(speed), 6),
                "matched_iou": round(float(best_iou), 6),
            }
            output_dets.append(enriched)
            active[best_id] = {
                "bbox_xyxy": bbox,
                "center_xy": (cx, cy),
                "timestamp_sec": timestamp,
                "last_frame_index": frame_index,
                "class_name": class_name,
            }
            assigned_tracks.add(best_id)

        stale = [track_id for track_id, state in active.items() if frame_index - int(state["last_frame_index"]) > max_age_frames]
        for track_id in stale:
            active.pop(track_id, None)
        tracked_frames.append({**frame, "detections": output_dets})
    return tracked_frames


def _track_histories(frames: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    histories: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        timestamp = float(frame.get("timestamp_sec", 0.0))
        frame_index = int(frame.get("frame_index", 0))
        for det in frame.get("detections", []):
            if "track_id" not in det:
                continue
            histories[int(det["track_id"])].append({**det, "timestamp_sec": timestamp, "frame_index": frame_index})
    for items in histories.values():
        items.sort(key=lambda item: (float(item["timestamp_sec"]), int(item["frame_index"])))
    return histories


def _make_segment(
    *,
    video_id: str,
    start: float,
    end: float,
    reason: str,
    score: float,
    track_ids: set[int],
    evidence: dict[str, Any],
    pre_window_sec: float,
    post_window_sec: float,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "segment_start_sec": round(max(0.0, start - pre_window_sec), 3),
        "segment_end_sec": round(max(end, start) + post_window_sec, 3),
        "peak_time_sec": round((start + end) / 2.0, 3),
        "candidate_score": round(float(score), 6),
        "trigger_reasons": [reason],
        "evidence_track_ids": sorted(int(t) for t in track_ids),
        "send_to_video_model": True,
        "evidence": evidence,
    }


def _merge_segments(segments: list[dict[str, Any]], max_gap_sec: float = 1.0) -> list[dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (float(item["segment_start_sec"]), float(item["segment_end_sec"])))
    merged: list[dict[str, Any]] = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        if float(seg["segment_start_sec"]) <= float(last["segment_end_sec"]) + max_gap_sec:
            last["segment_end_sec"] = round(max(float(last["segment_end_sec"]), float(seg["segment_end_sec"])), 3)
            last["peak_time_sec"] = round(max(float(last["peak_time_sec"]), float(seg["peak_time_sec"])), 3)
            last["candidate_score"] = round(max(float(last["candidate_score"]), float(seg["candidate_score"])), 6)
            last["trigger_reasons"] = sorted(set(last.get("trigger_reasons", [])) | set(seg.get("trigger_reasons", [])))
            last["evidence_track_ids"] = sorted(set(last.get("evidence_track_ids", [])) | set(seg.get("evidence_track_ids", [])))
            last.setdefault("evidence", {}).setdefault("merged_events", []).append(seg.get("evidence", {}))
        else:
            merged.append(seg)
    return merged


def build_track_candidate_segments(
    frames: list[dict[str, Any]],
    *,
    video_id: str,
    speed_drop_ratio: float = 0.35,
    min_speed_before_drop: float = 12.0,
    low_speed_px_per_sec: float = 2.0,
    abnormal_stop_sec: float = 2.0,
    overlap_iou: float = 0.35,
    queue_count: int = 8,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 4.0,
    max_segments: int = 8,
) -> list[dict[str, Any]]:
    """Build candidate segments from track dynamics.

    The output is deliberately conservative evidence for VideoMAE screening,
    not a final accident classification.
    """
    segments: list[dict[str, Any]] = []
    histories = _track_histories(frames)

    for track_id, history in histories.items():
        if len(history) < 2:
            continue
        low_start: float | None = None
        prev_speed = float(history[0].get("speed_px_per_sec", 0.0))
        for item in history[1:]:
            ts = float(item["timestamp_sec"])
            speed = float(item.get("speed_px_per_sec", 0.0))
            if prev_speed >= min_speed_before_drop and speed <= prev_speed * speed_drop_ratio:
                score = min(5.0, prev_speed / max(speed, 1.0))
                segments.append(_make_segment(
                    video_id=video_id,
                    start=ts,
                    end=ts,
                    reason="speed_drop",
                    score=score,
                    track_ids={track_id},
                    evidence={"track_id": track_id, "prev_speed_px_per_sec": prev_speed, "speed_px_per_sec": speed},
                    pre_window_sec=pre_window_sec,
                    post_window_sec=post_window_sec,
                ))
            if speed <= low_speed_px_per_sec:
                low_start = ts if low_start is None else low_start
                if ts - low_start >= abnormal_stop_sec:
                    segments.append(_make_segment(
                        video_id=video_id,
                        start=low_start,
                        end=ts,
                        reason="abnormal_stop",
                        score=1.0 + (ts - low_start) / max(abnormal_stop_sec, 1e-6),
                        track_ids={track_id},
                        evidence={"track_id": track_id, "stop_duration_sec": round(ts - low_start, 3), "low_speed_px_per_sec": low_speed_px_per_sec},
                        pre_window_sec=pre_window_sec,
                        post_window_sec=post_window_sec,
                    ))
                    low_start = None
            else:
                low_start = None
            prev_speed = speed

    for frame in frames:
        detections = [det for det in frame.get("detections", []) if "track_id" in det]
        ts = float(frame.get("timestamp_sec", 0.0))
        if len(detections) >= queue_count:
            segments.append(_make_segment(
                video_id=video_id,
                start=ts,
                end=ts,
                reason="queue_growth",
                score=len(detections) / max(queue_count, 1),
                track_ids={int(det["track_id"]) for det in detections},
                evidence={"frame_index": int(frame.get("frame_index", 0)), "object_count": len(detections)},
                pre_window_sec=pre_window_sec,
                post_window_sec=post_window_sec,
            ))
        for idx, det_a in enumerate(detections):
            for det_b in detections[idx + 1:]:
                score = iou_xyxy(det_a.get("bbox_xyxy", []), det_b.get("bbox_xyxy", []))
                if score >= overlap_iou:
                    segments.append(_make_segment(
                        video_id=video_id,
                        start=ts,
                        end=ts,
                        reason="bbox_overlap",
                        score=score,
                        track_ids={int(det_a["track_id"]), int(det_b["track_id"])},
                        evidence={"frame_index": int(frame.get("frame_index", 0)), "overlap_iou": round(score, 6)},
                        pre_window_sec=pre_window_sec,
                        post_window_sec=post_window_sec,
                    ))
    merged = _merge_segments(segments)
    merged.sort(key=lambda item: float(item.get("candidate_score", 0.0)), reverse=True)
    selected = merged[:max_segments]
    selected.sort(key=lambda item: float(item["segment_start_sec"]))
    return selected


def select_evidence_detections(
    frames: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    track_ids: set[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_sec", 0.0))
        if ts < start_sec or ts > end_sec:
            continue
        detections = []
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if track_ids and int(tid) not in track_ids:
                continue
            detections.append(det)
        if detections:
            selected[int(frame.get("frame_index", 0))] = detections
    return selected


LEGACY_TRAJECTORY_REASON_MAP = {
    0: ("abnormal_stop", "weiting_ids"),
    1: ("trajectory_conflict", "trajectory_anomaly_ids"),
    3: ("warning_sign_missing", "warning_sign_ids"),
    4: ("low_speed", "low_speed_ids"),
    5: ("legacy_accident_suspect", "legacy_accident_ids"),
    6: ("queue_growth", "congestion_ids"),
}


def merge_candidate_segments(segments: list[dict[str, Any]], max_gap_sec: float = 1.0) -> list[dict[str, Any]]:
    return _merge_segments(segments, max_gap_sec=max_gap_sec)


def track_time_ranges(frames: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    ranges: dict[int, dict[str, float]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_sec", 0.0))
        for det in frame.get("detections", []):
            if "track_id" not in det:
                continue
            tid = int(det["track_id"])
            item = ranges.setdefault(tid, {"start_sec": ts, "end_sec": ts})
            item["start_sec"] = min(float(item["start_sec"]), ts)
            item["end_sec"] = max(float(item["end_sec"]), ts)
    return ranges


def build_legacy_candidate_segments(
    frames: list[dict[str, Any]],
    trajectory_events: dict[str, Any],
    *,
    video_id: str,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 4.0,
    max_segments: int = 8,
) -> list[dict[str, Any]]:
    ranges = track_time_ranges(frames)
    all_times = [float(frame.get("timestamp_sec", 0.0)) for frame in frames]
    fallback_start = min(all_times) if all_times else 0.0
    fallback_end = max(all_times) if all_times else 0.0
    segments: list[dict[str, Any]] = []
    for event in trajectory_events.get("events", []):
        reason = str(event.get("reason", "unknown"))
        track_ids = {int(tid) for tid in event.get("evidence_track_ids", []) if tid is not None}
        starts = [ranges[tid]["start_sec"] for tid in track_ids if tid in ranges]
        ends = [ranges[tid]["end_sec"] for tid in track_ids if tid in ranges]
        start = min(starts) if starts else fallback_start
        end = max(ends) if ends else fallback_end
        score = float(event.get("candidate_score", 1.0))
        segments.append(_make_segment(
            video_id=video_id,
            start=start,
            end=end,
            reason=reason,
            score=score,
            track_ids=track_ids,
            evidence={
                "source": "legacy_trackVehicleInSeqpre",
                "legacy_flag_index": event.get("legacy_flag_index"),
                "legacy_event_type": event.get("legacy_event_type"),
                "legacy_flag_value": event.get("legacy_flag_value"),
            },
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
        ))
    merged = _merge_segments(segments)
    merged.sort(key=lambda item: float(item.get("candidate_score", 0.0)), reverse=True)
    selected = merged[:max_segments]
    selected.sort(key=lambda item: float(item["segment_start_sec"]))
    return selected


def build_final_accident_events(
    predictions: list[dict[str, Any]],
    *,
    video_id: str,
    video_path: str,
    threshold: float,
    checkpoint: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        score = float(pred.get("accident_score", 0.0))
        if score < threshold:
            continue
        events.append({
            "event_id": f"{video_id}_accident_{idx:03d}",
            "event_type": "traffic_accident",
            "video_id": video_id,
            "video_path": video_path,
            "segment_start_sec": float(pred.get("segment_start_sec", 0.0)),
            "segment_end_sec": float(pred.get("segment_end_sec", 0.0)),
            "accident_score": score,
            "threshold": float(threshold),
            "decision": "accident",
            "video_model": "VideoMAE",
            "video_model_checkpoint": checkpoint,
            "trigger_reasons": list(pred.get("trigger_reasons", [])),
            "evidence_track_ids": [int(tid) for tid in pred.get("evidence_track_ids", [])],
            "candidate_id": int(pred.get("candidate_id", idx)),
            "clip_path": pred.get("clip_path"),
            "notes": "YOLO/Track evidence supports candidate selection and visualization only; VideoMAE score gates the final accident decision.",
        })
    return events


def build_accident_evidence_tracks(
    predictions: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Return frame-level suspect vehicle evidence for VideoMAE-positive candidates."""
    evidence_rows: list[dict[str, Any]] = []
    positive = [pred for pred in predictions if float(pred.get("accident_score", 0.0)) >= threshold]
    for pred in positive:
        start_sec = float(pred.get("segment_start_sec", 0.0))
        end_sec = float(pred.get("segment_end_sec", 0.0))
        track_ids = {int(tid) for tid in pred.get("evidence_track_ids", []) if tid is not None}
        if not track_ids:
            continue
        for frame in frames:
            ts = float(frame.get("timestamp_sec", 0.0))
            if ts < start_sec or ts > end_sec:
                continue
            frame_index = int(frame.get("frame_index", 0))
            for det in frame.get("detections", []):
                tid_raw = det.get("track_id")
                if tid_raw is None:
                    continue
                tid = int(tid_raw)
                if tid not in track_ids:
                    continue
                evidence_rows.append({
                    "label": "SUSPECT_ACCIDENT_VEHICLE",
                    "candidate_id": int(pred.get("candidate_id", 0)),
                    "frame_index": frame_index,
                    "timestamp_sec": ts,
                    "track_id": tid,
                    "class_name": str(det.get("class_name", "")),
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox_xyxy": [float(v) for v in det.get("bbox_xyxy", [])],
                    "accident_score": float(pred.get("accident_score", 0.0)),
                    "threshold": float(threshold),
                    "segment_start_sec": start_sec,
                    "segment_end_sec": end_sec,
                    "trigger_reasons": list(pred.get("trigger_reasons", [])),
                    "notes": "Suspect evidence vehicle from YOLO/Track candidate evidence; not box-level accident ground truth.",
                })
    return evidence_rows
