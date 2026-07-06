"""Video loading utilities for baseline training and inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def read_video_metadata(video_path: str | Path) -> dict:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "video_path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 and frame_count else None,
    }


def load_video_clip_tensor(
    video_path: str | Path,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    num_frames: int = 16,
    size: int = 112,
) -> torch.Tensor:
    """Load a clip as a C x T x H x W float tensor in [0, 1]."""
    path = Path(video_path)
    metadata = read_video_metadata(path)
    duration = metadata.get("duration_sec") or max(start_sec + 1.0, 1.0)
    safe_end = end_sec if end_sec is not None else duration
    safe_end = max(safe_end, start_sec + 1e-3)
    if duration:
        safe_end = min(safe_end, duration)
        start_sec = min(max(0.0, start_sec), max(0.0, duration - 1e-3))
    target_times = np.linspace(start_sec, safe_end, num_frames, endpoint=False)

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None
    for timestamp in target_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            if last_frame is None:
                continue
            frame = last_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        last_frame = rgb
        frames.append(rgb)
    capture.release()
    if not frames:
        raise ValueError(f"Could not read any frame from video: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    array = np.stack(frames[:num_frames], axis=0).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return tensor
