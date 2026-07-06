"""Inference pipeline combining candidate triggering and R3D-18 scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch

from traffic_accident_rnd.model import build_r3d18, load_checkpoint, select_device
from traffic_accident_rnd.trigger import detect_candidate_segments
from traffic_accident_rnd.video_io import load_video_clip_tensor


def summarize_track_file(track_path: str | Path) -> dict:
    path = Path(track_path)
    class_counts: Counter[str] = Counter()
    frame_count = 0
    detection_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame_count += 1
            record = json.loads(line)
            detections = record.get("detections", [])
            detection_count += len(detections)
            class_counts.update(det.get("class_name", "unknown") for det in detections)
    return {"frame_count": frame_count, "detection_count": detection_count, "class_counts": dict(class_counts)}


def score_segments_with_model(
    video_path: str | Path,
    segments: Sequence[dict],
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    pretrained: bool = False,
    num_frames: int = 16,
    size: int = 112,
) -> list[float]:
    selected = select_device(device)
    model = build_r3d18(pretrained=pretrained).to(selected)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=selected)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for segment in segments:
            clip = load_video_clip_tensor(
                video_path,
                start_sec=float(segment["segment_start_sec"]),
                end_sec=float(segment["segment_end_sec"]),
                num_frames=num_frames,
                size=size,
            ).unsqueeze(0).to(selected)
            logits = model(clip)
            probability = torch.softmax(logits, dim=1)[0, 1].detach().cpu().item()
            scores.append(round(float(probability), 6))
    return scores


def build_prediction_result(
    *,
    video_path: str | Path,
    segments: Sequence[dict],
    scores: Sequence[float],
    track_summary: dict | None = None,
    output_path: str | Path | None = None,
) -> dict:
    predictions = []
    for segment, score in zip(segments, scores):
        record = dict(segment)
        record["accident_score"] = round(float(score), 6)
        predictions.append(record)
    result = {
        "video_path": str(video_path),
        "candidate_count": len(predictions),
        "predictions": predictions,
        "track_summary": track_summary,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def predict_video(
    video_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    track_path: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold_z: float = 2.5,
    sample_fps: float = 4.0,
    device: str = "auto",
    pretrained: bool = False,
) -> dict:
    segments = detect_candidate_segments(video_path, threshold_z=threshold_z, sample_fps=sample_fps)
    scores = score_segments_with_model(video_path, segments, checkpoint_path=checkpoint_path, device=device, pretrained=pretrained)
    track_summary = summarize_track_file(track_path) if track_path else None
    return build_prediction_result(video_path=video_path, segments=segments, scores=scores, track_summary=track_summary, output_path=output_path)
