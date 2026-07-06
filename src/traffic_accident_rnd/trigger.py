"""Frame-difference accident candidate trigger for fixed traffic cameras."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def _z_scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def detect_segments_from_scores(
    times_sec: Sequence[float],
    scores: Sequence[float],
    *,
    threshold_z: float = 2.5,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 4.0,
    min_gap_sec: float = 1.0,
    max_segments: int = 5,
    video_id: str | None = None,
) -> list[dict]:
    """Convert motion scores into candidate accident segments."""
    if len(times_sec) != len(scores):
        raise ValueError("times_sec and scores must have the same length")
    if not scores:
        return []

    zscores = _z_scores(scores)
    peak_indices = [idx for idx, zscore in enumerate(zscores) if float(zscore) >= threshold_z]
    if not peak_indices:
        return []

    groups: list[list[int]] = []
    current: list[int] = []
    for idx in peak_indices:
        if not current or float(times_sec[idx]) - float(times_sec[current[-1]]) <= min_gap_sec:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    if current:
        groups.append(current)

    segments: list[dict] = []
    score_mean = float(np.mean(np.asarray(scores, dtype=np.float32)))
    score_std = float(np.std(np.asarray(scores, dtype=np.float32)))
    for group in groups:
        peak_idx = max(group, key=lambda i: float(zscores[i]))
        start = max(0.0, float(times_sec[group[0]]) - pre_window_sec)
        end = float(times_sec[group[-1]]) + post_window_sec
        segment = {
            "video_id": video_id,
            "segment_start_sec": round(start, 3),
            "segment_end_sec": round(end, 3),
            "peak_time_sec": round(float(times_sec[peak_idx]), 3),
            "trigger_score": round(float(zscores[peak_idx]), 6),
            "evidence": {
                "method": "frame_diff_zscore",
                "peak_count": len(group),
                "peak_indices": group,
                "raw_peak_score": round(float(scores[peak_idx]), 6),
                "score_mean": round(score_mean, 6),
                "score_std": round(score_std, 6),
                "threshold_z": threshold_z,
            },
        }
        segments.append(segment)

    segments.sort(key=lambda item: item["trigger_score"], reverse=True)
    selected = segments[:max_segments]
    selected.sort(key=lambda item: item["segment_start_sec"])
    return selected


def compute_frame_motion_scores(
    video_path: str | Path,
    *,
    sample_fps: float = 4.0,
    resize_width: int = 160,
    resize_height: int = 90,
    max_frames: int | None = None,
) -> tuple[list[float], list[float], dict]:
    """Read a video and compute mean absolute frame-difference scores."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or sample_fps
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(native_fps / sample_fps))) if sample_fps > 0 else 1
    times: list[float] = []
    scores: list[float] = []
    previous = None
    frame_index = 0
    used_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            if previous is not None:
                diff = cv2.absdiff(gray, previous)
                scores.append(float(np.mean(diff)))
                times.append(frame_index / native_fps)
            previous = gray
            used_frames += 1
            if max_frames is not None and used_frames >= max_frames:
                break
        frame_index += 1
    capture.release()

    metadata = {
        "video_path": str(path),
        "native_fps": native_fps,
        "frame_count": frame_count,
        "sample_fps": sample_fps,
        "sampled_frame_count": used_frames,
        "duration_sec": frame_count / native_fps if native_fps > 0 and frame_count else None,
    }
    return times, scores, metadata


def detect_candidate_segments(
    video_path: str | Path,
    *,
    threshold_z: float = 2.5,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 4.0,
    min_gap_sec: float = 1.0,
    max_segments: int = 5,
    sample_fps: float = 4.0,
    resize_width: int = 160,
    resize_height: int = 90,
    video_id: str | None = None,
) -> list[dict]:
    times, scores, metadata = compute_frame_motion_scores(
        video_path,
        sample_fps=sample_fps,
        resize_width=resize_width,
        resize_height=resize_height,
    )
    segments = detect_segments_from_scores(
        times,
        scores,
        threshold_z=threshold_z,
        pre_window_sec=pre_window_sec,
        post_window_sec=post_window_sec,
        min_gap_sec=min_gap_sec,
        max_segments=max_segments,
        video_id=video_id or Path(video_path).stem,
    )
    for segment in segments:
        segment["evidence"]["video_metadata"] = metadata
    return segments


def write_candidate_segments(segments: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment, ensure_ascii=False, sort_keys=True) + "\n")
