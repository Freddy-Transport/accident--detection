"""Utilities for the YOLO/Track -> candidate -> VideoMAE cascade."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
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


def _bbox_diagonal(bbox: list[float] | tuple[float, ...]) -> float:
    if len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return math.hypot(x2 - x1, y2 - y1)


def _velocity(current: dict[str, Any], previous: dict[str, Any] | None) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0
    dt = float(current["timestamp_sec"]) - float(previous["timestamp_sec"])
    if dt <= 1e-6:
        return 0.0, 0.0
    cx, cy = bbox_center_xyxy(current.get("bbox_xyxy", []))
    px, py = bbox_center_xyxy(previous.get("bbox_xyxy", []))
    return (cx - px) / dt, (cy - py) / dt


def _can_emit(
    emitted: dict[tuple[str, tuple[int, ...]], float],
    reason: str,
    track_ids: set[int],
    timestamp: float,
    cooldown_sec: float,
) -> bool:
    key = (reason, tuple(sorted(track_ids)))
    previous = emitted.get(key)
    if previous is not None and timestamp - previous < cooldown_sec:
        return False
    emitted[key] = timestamp
    return True


def _smooth_track_speeds(history: list[dict[str, Any]], radius: int = 1) -> list[float]:
    raw = [max(0.0, float(item.get("speed_px_per_sec", 0.0))) for item in history]
    return [float(median(raw[max(0, idx - radius): min(len(raw), idx + radius + 1)]))
            for idx in range(len(raw))]


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
    trigger_time_sec: float | None = None,
) -> dict[str, Any]:
    trigger_time = float(trigger_time_sec if trigger_time_sec is not None else (start + end) / 2.0)
    return {
        "video_id": video_id,
        "segment_start_sec": round(max(0.0, start - pre_window_sec), 3),
        "segment_end_sec": round(max(end, start) + post_window_sec, 3),
        "peak_time_sec": round(trigger_time, 3),
        "trigger_time_sec": round(trigger_time, 3),
        "candidate_score": round(float(score), 6),
        "trigger_reasons": [reason],
        "evidence_track_ids": sorted(int(t) for t in track_ids),
        "primary_evidence_track_ids": sorted(int(t) for t in track_ids),
        "send_to_video_model": True,
        "evidence": evidence,
    }


def _merge_segments(
    segments: list[dict[str, Any]], max_gap_sec: float = 1.0,
    max_duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (float(item["segment_start_sec"]), float(item["segment_end_sec"])))
    merged: list[dict[str, Any]] = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        combined_end = max(float(last["segment_end_sec"]), float(seg["segment_end_sec"]))
        combined_duration = combined_end - float(last["segment_start_sec"])
        can_merge = float(seg["segment_start_sec"]) <= float(last["segment_end_sec"]) + max_gap_sec
        last_ids = set(last.get("evidence_track_ids", []))
        seg_ids = set(seg.get("evidence_track_ids", []))
        if last_ids and seg_ids and not last_ids.intersection(seg_ids):
            can_merge = False
        if max_duration_sec is not None and combined_duration > max_duration_sec:
            can_merge = False
        if can_merge:
            last["segment_end_sec"] = round(max(float(last["segment_end_sec"]), float(seg["segment_end_sec"])), 3)
            if float(seg["candidate_score"]) > float(last["candidate_score"]):
                last["peak_time_sec"] = float(seg["peak_time_sec"])
                last["trigger_time_sec"] = float(seg.get("trigger_time_sec", seg["peak_time_sec"]))
                last["primary_evidence_track_ids"] = list(
                    seg.get("primary_evidence_track_ids", seg.get("evidence_track_ids", [])))
            last["candidate_score"] = round(max(float(last["candidate_score"]), float(seg["candidate_score"])), 6)
            last["trigger_reasons"] = sorted(set(last.get("trigger_reasons", [])) | set(seg.get("trigger_reasons", [])))
            last["evidence_track_ids"] = sorted(set(last.get("evidence_track_ids", [])) | set(seg.get("evidence_track_ids", [])))
            last.setdefault("evidence", {}).setdefault("merged_events", []).append(seg.get("evidence", {}))
        else:
            merged.append(seg)
    return merged


def _build_track_candidate_segments_v0(
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


def _motion_rule_segments(
    histories: dict[int, list[dict[str, Any]]], *, video_id: str,
    speed_drop_ratio: float, min_speed_before_drop: float,
    low_speed_px_per_sec: float, abnormal_stop_sec: float,
    pre_window_sec: float, post_window_sec: float,
    cooldown_sec: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    emitted: dict[tuple[str, tuple[int, ...]], float] = {}
    for track_id, history in histories.items():
        if len(history) < 4:
            continue
        speeds = _smooth_track_speeds(history)
        moving_seen = False
        low_start: float | None = None
        stop_emitted = False
        for idx, item in enumerate(history):
            ts = float(item["timestamp_sec"])
            speed = speeds[idx]
            moving_seen = moving_seen or speed >= min_speed_before_drop
            if idx >= 4:
                before = float(median(speeds[max(0, idx - 4):idx]))
                after = float(median(speeds[idx:min(len(speeds), idx + 2)]))
                if (before >= min_speed_before_drop and after <= before * speed_drop_ratio
                        and _can_emit(emitted, "speed_drop", {track_id}, ts, cooldown_sec)):
                    segments.append(_make_segment(
                        video_id=video_id, start=ts, end=ts, reason="speed_drop",
                        score=min(5.0, before / max(after, 1.0)), track_ids={track_id},
                        evidence={"track_id": track_id, "speed_before_px_per_sec": round(before, 3),
                                  "speed_after_px_per_sec": round(after, 3)},
                        pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                        trigger_time_sec=ts,
                    ))
            if moving_seen and speed <= low_speed_px_per_sec:
                low_start = ts if low_start is None else low_start
                duration = ts - low_start
                if (duration >= abnormal_stop_sec and not stop_emitted
                        and _can_emit(emitted, "abnormal_stop", {track_id}, ts, cooldown_sec)):
                    segments.append(_make_segment(
                        video_id=video_id, start=low_start, end=ts, reason="abnormal_stop",
                        score=1.0 + duration / max(abnormal_stop_sec, 1e-6), track_ids={track_id},
                        evidence={"track_id": track_id, "stop_start_sec": round(low_start, 3),
                                  "stop_duration_sec": round(duration, 3)},
                        pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                        trigger_time_sec=ts,
                    ))
                    stop_emitted = True
            else:
                low_start = None
                stop_emitted = False
    return segments


def _interaction_rule_segments(
    frames: list[dict[str, Any]], *, video_id: str,
    overlap_iou: float, overlap_persistence_sec: float,
    conflict_horizon_sec: float, min_conflict_speed: float,
    queue_count: int, queue_growth_count: int, queue_window_sec: float,
    pre_window_sec: float, post_window_sec: float, cooldown_sec: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    emitted: dict[tuple[str, tuple[int, ...]], float] = {}
    previous: dict[int, dict[str, Any]] = {}
    overlap_state: dict[tuple[int, int], tuple[float, float]] = {}
    count_history: deque[tuple[float, int]] = deque()

    for frame in sorted(frames, key=lambda row: int(row.get("frame_index", 0))):
        ts = float(frame.get("timestamp_sec", 0.0))
        frame_index = int(frame.get("frame_index", 0))
        detections = [dict(det, timestamp_sec=ts) for det in frame.get("detections", [])
                      if det.get("track_id") is not None and len(det.get("bbox_xyxy", [])) == 4]
        current = {int(det["track_id"]): det for det in detections}
        velocities = {track_id: _velocity(det, previous.get(track_id))
                      for track_id, det in current.items()}

        count_history.append((ts, len(detections)))
        while count_history and ts - count_history[0][0] > queue_window_sec:
            count_history.popleft()
        baseline_count = min((count for _, count in count_history), default=len(detections))
        queue_ids = set(current)
        if (len(detections) >= queue_count and len(detections) - baseline_count >= queue_growth_count
                and queue_ids and _can_emit(emitted, "queue_growth", queue_ids, ts, cooldown_sec)):
            segments.append(_make_segment(
                video_id=video_id, start=ts, end=ts, reason="queue_growth",
                score=1.0 + (len(detections) - baseline_count) / max(queue_growth_count, 1),
                track_ids=queue_ids,
                evidence={"frame_index": frame_index, "object_count": len(detections),
                          "baseline_count": baseline_count, "window_sec": queue_window_sec},
                pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                trigger_time_sec=ts,
            ))

        active_overlap_pairs: set[tuple[int, int]] = set()
        for idx, det_a in enumerate(detections):
            tid_a = int(det_a["track_id"])
            for det_b in detections[idx + 1:]:
                tid_b = int(det_b["track_id"])
                pair = tuple(sorted((tid_a, tid_b)))
                track_ids = set(pair)
                overlap = iou_xyxy(det_a["bbox_xyxy"], det_b["bbox_xyxy"])
                if overlap >= overlap_iou:
                    active_overlap_pairs.add(pair)
                    start, best = overlap_state.get(pair, (ts, overlap))
                    overlap_state[pair] = (start, max(best, overlap))
                    if (ts - start >= overlap_persistence_sec
                            and _can_emit(emitted, "bbox_overlap", track_ids, ts, cooldown_sec)):
                        segments.append(_make_segment(
                            video_id=video_id, start=start, end=ts, reason="bbox_overlap",
                            score=1.0 + max(best, overlap), track_ids=track_ids,
                            evidence={"frame_index": frame_index, "overlap_iou": round(max(best, overlap), 4),
                                      "persistence_sec": round(ts - start, 3)},
                            pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                            trigger_time_sec=ts,
                        ))

                va, vb = velocities.get(tid_a, (0.0, 0.0)), velocities.get(tid_b, (0.0, 0.0))
                speed_a, speed_b = math.hypot(*va), math.hypot(*vb)
                if speed_a < min_conflict_speed or speed_b < min_conflict_speed:
                    continue
                ca, cb = bbox_center_xyxy(det_a["bbox_xyxy"]), bbox_center_xyxy(det_b["bbox_xyxy"])
                rx, ry = ca[0] - cb[0], ca[1] - cb[1]
                rvx, rvy = va[0] - vb[0], va[1] - vb[1]
                rv_norm_sq = rvx * rvx + rvy * rvy
                if rv_norm_sq <= 1e-6:
                    continue
                ttc = -(rx * rvx + ry * rvy) / rv_norm_sq
                closest = math.hypot(rx + rvx * ttc, ry + rvy * ttc)
                distance_limit = max(12.0, 0.35 * min(_bbox_diagonal(det_a["bbox_xyxy"]),
                                                     _bbox_diagonal(det_b["bbox_xyxy"])))
                if (0.0 < ttc <= conflict_horizon_sec and closest <= distance_limit
                        and _can_emit(emitted, "trajectory_conflict", track_ids, ts, cooldown_sec)):
                    segments.append(_make_segment(
                        video_id=video_id, start=ts, end=ts, reason="trajectory_conflict",
                        score=min(5.0, 1.0 + (conflict_horizon_sec - ttc) + distance_limit / max(closest, 1.0)),
                        track_ids=track_ids,
                        evidence={"frame_index": frame_index, "ttc_sec": round(ttc, 3),
                                  "predicted_distance_px": round(closest, 3),
                                  "distance_limit_px": round(distance_limit, 3)},
                        pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                        trigger_time_sec=ts,
                    ))
        overlap_state = {pair: state for pair, state in overlap_state.items() if pair in active_overlap_pairs}
        previous.update(current)
    return segments


def build_track_candidate_segments(
    frames: list[dict[str, Any]], *, video_id: str,
    speed_drop_ratio: float = 0.4,
    min_speed_before_drop: float = 20.0,
    low_speed_px_per_sec: float = 3.0,
    abnormal_stop_sec: float = 1.5,
    overlap_iou: float = 0.2,
    overlap_persistence_sec: float = 0.15,
    conflict_horizon_sec: float = 2.0,
    min_conflict_speed: float = 8.0,
    queue_count: int = 8,
    queue_growth_count: int = 3,
    queue_window_sec: float = 2.0,
    cooldown_sec: float = 2.0,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 4.0,
    max_segment_duration_sec: float = 10.0,
    max_segments: int = 8,
) -> list[dict[str, Any]]:
    """Generate timestamped candidates from tracks without annotation access."""
    histories = _track_histories(frames)
    segments = _motion_rule_segments(
        histories, video_id=video_id,
        speed_drop_ratio=speed_drop_ratio,
        min_speed_before_drop=min_speed_before_drop,
        low_speed_px_per_sec=low_speed_px_per_sec,
        abnormal_stop_sec=abnormal_stop_sec,
        pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
        cooldown_sec=cooldown_sec,
    )
    segments.extend(_interaction_rule_segments(
        frames, video_id=video_id,
        overlap_iou=overlap_iou,
        overlap_persistence_sec=overlap_persistence_sec,
        conflict_horizon_sec=conflict_horizon_sec,
        min_conflict_speed=min_conflict_speed,
        queue_count=queue_count,
        queue_growth_count=queue_growth_count,
        queue_window_sec=queue_window_sec,
        pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
        cooldown_sec=cooldown_sec,
    ))
    merged = _merge_segments(
        segments, max_gap_sec=0.25,
        max_duration_sec=max_segment_duration_sec,
    )
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
    positive_events: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        score = float(pred.get("accident_score", 0.0))
        if score < threshold:
            continue
        positive_events.append({
            "event_id": f"{video_id}_accident_{idx:03d}",
            "event_type": "traffic_accident",
            "video_id": video_id,
            "video_path": video_path,
            "segment_start_sec": float(pred.get("segment_start_sec", 0.0)),
            "segment_end_sec": float(pred.get("segment_end_sec", 0.0)),
            "detected_time_sec": float(pred.get("trigger_time_sec", pred.get("peak_time_sec", 0.0))),
            "candidate_score": float(pred.get("candidate_score", 0.0)),
            "accident_score": score,
            "threshold": float(threshold),
            "decision": "accident",
            "video_model": "VideoMAE",
            "video_model_checkpoint": checkpoint,
            "trigger_reasons": list(pred.get("trigger_reasons", [])),
            "evidence_track_ids": [int(tid) for tid in pred.get(
                "primary_evidence_track_ids", pred.get("evidence_track_ids", []))],
            "all_trigger_track_ids": [int(tid) for tid in pred.get("evidence_track_ids", [])],
            "candidate_id": int(pred.get("candidate_id", idx)),
            "clip_path": pred.get("clip_path"),
            "notes": "YOLO/Track evidence supports candidate selection and visualization only; VideoMAE score gates the final accident decision.",
        })

    # Candidate clips intentionally overlap around nearby track triggers. Collapse
    # overlapping VideoMAE-positive windows into one incident and retain the
    # strongest candidate as the event time and suspect-vehicle source.
    positive_events.sort(key=lambda item: (item["segment_start_sec"], item["segment_end_sec"]))
    clusters: list[list[dict[str, Any]]] = []
    cluster_end = float("-inf")
    for event in positive_events:
        start = float(event["segment_start_sec"])
        if not clusters or start > cluster_end:
            clusters.append([event])
            cluster_end = float(event["segment_end_sec"])
        else:
            clusters[-1].append(event)
            cluster_end = max(cluster_end, float(event["segment_end_sec"]))

    events: list[dict[str, Any]] = []
    for event_idx, cluster in enumerate(clusters):
        representative = max(
            cluster,
            key=lambda item: (float(item["accident_score"]), float(item["candidate_score"])),
        )
        event = dict(representative)
        event["event_id"] = f"{video_id}_accident_{event_idx:03d}"
        event["supporting_candidate_ids"] = sorted(int(item["candidate_id"]) for item in cluster)
        event["supporting_candidate_count"] = len(cluster)
        event["supporting_span_start_sec"] = min(float(item["segment_start_sec"]) for item in cluster)
        event["supporting_span_end_sec"] = max(float(item["segment_end_sec"]) for item in cluster)
        event["all_trigger_track_ids"] = sorted({
            int(track_id)
            for item in cluster
            for track_id in item.get("all_trigger_track_ids", [])
        })
        event["all_trigger_reasons"] = sorted({
            str(reason)
            for item in cluster
            for reason in item.get("trigger_reasons", [])
        })
        events.append(event)
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
        primary_ids = pred.get("primary_evidence_track_ids", pred.get("evidence_track_ids", []))
        track_ids = {int(tid) for tid in primary_ids if tid is not None}
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
