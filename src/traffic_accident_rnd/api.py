"""FastAPI demo service for traffic accident early-discovery inference."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from traffic_accident_rnd.inference import predict_video, summarize_track_file
from traffic_accident_rnd.schemas import REQUIRED_MANIFEST_FIELDS, TRACK_FRAME_FIELDS, VALID_LABELS, VALID_SPLITS

PROJECT_ROOT = Path("/root/autodl-tmp/traffic_accident_rnd")

app = FastAPI(
    title="Traffic Accident Early Discovery MVP",
    version="0.1.0",
    description="Demo inference API for fixed urban road surveillance videos.",
)


class PredictVideoRequest(BaseModel):
    video_path: Path
    output_path: Path | None = Field(default=None)
    checkpoint_path: Path | None = Field(default=None)
    track_path: Path | None = Field(default=None)
    threshold_z: float = 2.5
    sample_fps: float = 4.0
    device: str = "auto"


class PredictTracksRequest(BaseModel):
    track_path: Path


def _git_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() or None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "manifest_required_fields": sorted(REQUIRED_MANIFEST_FIELDS),
        "manifest_valid_labels": sorted(VALID_LABELS),
        "manifest_valid_splits": sorted(VALID_SPLITS),
        "track_frame_required_fields": sorted(TRACK_FRAME_FIELDS),
        "track_detection_schema": {
            "track_id": "string|number|null",
            "class_name": "string",
            "confidence": "float in [0, 1]",
            "bbox_xyxy": "[x1, y1, x2, y2] with x2 > x1 and y2 > y1",
        },
    }


@app.post("/predict/video")
def predict_video_endpoint(request: PredictVideoRequest) -> dict[str, Any]:
    try:
        output = request.output_path or PROJECT_ROOT / "outputs" / "inference" / "api_prediction.json"
        return predict_video(
            video_path=str(request.video_path),
            checkpoint_path=str(request.checkpoint_path) if request.checkpoint_path else None,
            track_path=str(request.track_path) if request.track_path else None,
            output_path=output,
            threshold_z=request.threshold_z,
            sample_fps=request.sample_fps,
            device=request.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/tracks")
def predict_tracks_endpoint(request: PredictTracksRequest) -> dict[str, Any]:
    try:
        return summarize_track_file(request.track_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
