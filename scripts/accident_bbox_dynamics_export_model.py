#!/usr/bin/env python3
"""Run ACCIDENT bbox dynamics with exported YOLO formats such as ONNX.

The official bbox_dynamics.py assumes a PyTorch .pt model and calls YOLO.to().
Ultralytics exported formats support predict/track but receive the device through
inference calls instead. This wrapper keeps the official evaluation code in use
and only swaps the Tracker class before calling bbox_dynamics.main().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURISTIC_DIR = PROJECT_ROOT / "third_party" / "ACCIDENT" / "baselines" / "heuristic"
if not HEURISTIC_DIR.exists():
    raise SystemExit(f"Missing official ACCIDENT heuristic dir: {HEURISTIC_DIR}")

sys.path.insert(0, str(HEURISTIC_DIR))
import bbox_dynamics as official  # noqa: E402


class ExportAwareTracker(official.Tracker):
    def __init__(
        self,
        model_path: str,
        image_resolution: int,
        batch_size: int,
        confidence_threshold: float,
        cuda_device_id: int,
    ):
        self.batch_size = batch_size
        self.image_resolution = image_resolution
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = YOLO(model_path, task="detect")

        suffix = Path(model_path).suffix.lower()
        self.use_tracking = suffix == ".pt"
        if self.use_tracking:
            device = torch.device(f"cuda:{cuda_device_id}" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.inference_device: int | str = cuda_device_id if device.type == "cuda" else "cpu"
            print(f"Using YOLO PyTorch model: {model_path} at {device.type} device.")
        else:
            # The provided cardet.onnx has fixed batch=1 and CPUExecutionProvider is reproducible here.
            self.inference_device = "cpu"
            print(f"Using YOLO exported model: {model_path} with device={self.inference_device}.")

    def track(self, batches):
        bboxes = []
        frames_indices = []
        class_ids, track_ids, confidences = [], [], []
        try:
            for batch_index, batch in enumerate(batches):
                if self.use_tracking:
                    results = self.model.track(
                        batch,
                        imgsz=self.image_resolution,
                        verbose=False,
                        tracker="bytetrack.yaml",
                        persist=True,
                        conf=self.confidence_threshold,
                        device=self.inference_device,
                    )
                else:
                    results = self.model.predict(
                        batch,
                        imgsz=self.image_resolution,
                        verbose=False,
                        conf=self.confidence_threshold,
                        device=self.inference_device,
                    )

                for i, x in enumerate(results):
                    frame_index = batch_index * self.batch_size + i
                    frames_indices.append(frame_index)
                    boxes = x.boxes
                    if boxes is None or len(boxes) == 0:
                        bboxes.append([])
                        class_ids.append([])
                        track_ids.append([])
                        confidences.append([])
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()
                    valid = np.isfinite(xyxy).all(axis=1) & (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
                    if not valid.any():
                        bboxes.append([])
                        class_ids.append([])
                        track_ids.append([])
                        confidences.append([])
                        continue

                    valid_indices = np.flatnonzero(valid)
                    bboxes.append(xyxy[valid].tolist())
                    class_ids.append(boxes.cls.cpu().numpy().astype(int)[valid].tolist())
                    if getattr(boxes, "id", None) is not None:
                        track_ids.append(boxes.id.cpu().numpy()[valid].tolist())
                    else:
                        track_ids.append(valid_indices.astype(int).tolist())
                    confidences.append(boxes.conf.cpu().numpy()[valid].tolist())
        except Exception as exc:
            print(f"ERROR: {exc}")
        finally:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None)
            if trackers is not None:
                for tracker in trackers:
                    tracker.reset()

        return {
            "frames": frames_indices,
            "bboxes": bboxes,
            "class_ids": class_ids,
            "track_ids": track_ids,
            "confidences": confidences,
        }


def main() -> None:
    official.Tracker = ExportAwareTracker
    official.main()


if __name__ == "__main__":
    main()
