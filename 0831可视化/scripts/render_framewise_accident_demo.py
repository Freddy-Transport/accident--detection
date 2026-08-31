#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
GREEN = (70, 200, 90)
RED = (30, 30, 235)
ORANGE = (0, 165, 255)
WHITE = (245, 245, 245)
DARK = (18, 22, 28)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def normalized_region(metadata: dict[str, Any], width: int, height: int) -> list[float]:
    return [
        float(metadata["x1"]) * width,
        float(metadata["y1"]) * height,
        float(metadata["x2"]) * width,
        float(metadata["y2"]) * height,
    ]


def select_real_accident_tracks(
    rows: list[dict[str, Any]],
    region: list[float],
    event_time: float,
    limit: int = 2,
) -> tuple[list[int], dict[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    distances: dict[int, float] = defaultdict(lambda: float("inf"))
    rx = (region[0] + region[2]) / 2.0
    ry = (region[1] + region[3]) / 2.0
    rw = max(region[2] - region[0], 1.0)
    rh = max(region[3] - region[1], 1.0)
    for row in rows:
        ts = float(row.get("timestamp_sec", 0.0))
        if abs(ts - event_time) > 1.25:
            continue
        time_weight = max(0.1, 1.0 - abs(ts - event_time) / 1.25)
        for det in row.get("detections", []):
            if str(det.get("class_name", "")).lower() not in VEHICLE_CLASSES:
                continue
            box = [float(v) for v in det.get("bbox_xyxy", [])]
            if len(box) != 4 or det.get("track_id") is None:
                continue
            track_id = int(det["track_id"])
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            inside = region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
            overlap = iou(box, region)
            distance = ((cx - rx) / rw) ** 2 + ((cy - ry) / rh) ** 2
            distances[track_id] = min(distances[track_id], distance)
            scores[track_id] += time_weight * (
                4.0 * overlap + (1.0 if inside else 0.0) + 0.15 * float(det.get("confidence", 0.0))
            )
    ranked = sorted(scores, key=lambda tid: (scores[tid], -distances[tid]), reverse=True)
    if not ranked:
        ranked = sorted(distances, key=distances.get)
    return ranked[:limit], {tid: float(scores.get(tid, 0.0)) for tid in ranked[:limit]}


def track_points(rows: list[dict[str, Any]], selected: list[int]) -> dict[int, dict[int, list[float]]]:
    wanted = set(selected)
    result: dict[int, dict[int, list[float]]] = {tid: {} for tid in selected}
    for row in rows:
        frame_index = int(row.get("frame_index", 0))
        for det in row.get("detections", []):
            if det.get("track_id") is None:
                continue
            track_id = int(det["track_id"])
            box = det.get("bbox_xyxy", [])
            if track_id in wanted and len(box) == 4:
                result[track_id][frame_index] = [float(v) for v in box]
    return result


def synthetic_actor_points(
    annotation: dict[str, Any],
    actor_ids: list[int],
) -> dict[int, dict[int, list[float]]]:
    wanted = set(actor_ids)
    result: dict[int, dict[int, list[float]]] = {actor_id: {} for actor_id in actor_ids}
    for frame_index, row in enumerate(annotation.get("base", [])):
        for obj in row.get("objects", []):
            actor_id = int(obj.get("id", -1))
            box = obj.get("2d_bbox", [])
            if actor_id in wanted and len(box) == 2:
                result[actor_id][frame_index] = [
                    float(box[0][0]),
                    float(box[0][1]),
                    float(box[1][0]),
                    float(box[1][1]),
                ]
    return result


def interpolate_series(points: dict[int, list[float]], total_frames: int) -> np.ndarray | None:
    if not points:
        return None
    known = np.asarray(sorted(points), dtype=np.float32)
    values = np.asarray([points[int(idx)] for idx in known], dtype=np.float32)
    target = np.arange(total_frames, dtype=np.float32)
    coords = [np.interp(target, known, values[:, col]) for col in range(4)]
    return np.stack(coords, axis=1)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1100.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 5, y)
    cv2.rectangle(frame, (x, y - th - 7), (min(frame.shape[1] - 1, x + tw + 8), y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 2), font, scale, WHITE, thickness, cv2.LINE_AA)


def draw_box(frame: np.ndarray, box: np.ndarray | list[float], color: tuple[int, int, int], thickness: int) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def make_contact_sheet(frames: dict[int, np.ndarray], output: Path) -> None:
    if not frames:
        return
    tiles = []
    for idx in sorted(frames):
        frame = frames[idx]
        target_w = 480
        target_h = int(round(frame.shape[0] * target_w / frame.shape[1]))
        tile = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"frame {idx}", (12, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
        tiles.append(tile)
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tile_h = max(tile.shape[0] for tile in tiles)
    canvas = np.full((rows * tile_h, cols * 480, 3), 22, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // cols) * tile_h, (i % cols) * 480
        canvas[y:y + tile.shape[0], x:x + 480] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser(description="Render every-frame accident vehicle evidence demo.")
    parser.add_argument("--domain", choices=["real", "synthetic"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks-jsonl", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-evidence-json", required=True)
    parser.add_argument("--output-contact-sheet", required=True)
    parser.add_argument("--synthetic-annotation")
    args = parser.parse_args()

    tracks = read_jsonl(args.tracks_jsonl)
    predictions = read_json(args.prediction_json)
    prediction = predictions[0] if predictions else {}
    metadata = read_json(args.metadata_json)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if min(total_frames, width, height) <= 0:
        raise RuntimeError("invalid video metadata")

    event_time = float(metadata["demo_accident_time_sec"])
    region = normalized_region(metadata, width, height)
    association_scores: dict[int, float] = {}
    if args.domain == "synthetic":
        if not args.synthetic_annotation:
            raise ValueError("--synthetic-annotation is required for synthetic domain")
        actor_ids = [int(v) for v in metadata["collision_actor_ids"]]
        source_points = synthetic_actor_points(read_json(args.synthetic_annotation), actor_ids)
        selected_ids = actor_ids
        evidence_method = "CARLA collision actor IDs with per-frame 2D boxes"
    else:
        selected_ids, association_scores = select_real_accident_tracks(tracks, region, event_time)
        source_points = track_points(tracks, selected_ids)
        evidence_method = "YOLO/StrongSORT tracks associated with ACCIDENT annotated collision region"

    series = {item_id: interpolate_series(points, total_frames) for item_id, points in source_points.items()}
    series = {item_id: value for item_id, value in series.items() if value is not None}
    if not series:
        raise RuntimeError("no accident vehicle evidence boxes could be associated")

    rows_by_frame = {int(row.get("frame_index", 0)): row for row in tracks}
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {output_video}")

    score = float(prediction.get("accident_score", 0.0))
    threshold = float(prediction.get("threshold", 0.56))
    detected = score >= threshold
    event_frame = int(round(event_time * fps))
    capture_indices = {
        0,
        max(0, event_frame - int(round(fps))),
        max(0, min(total_frames - 1, event_frame)),
        max(0, min(total_frames - 1, event_frame + int(round(fps)))),
        total_frames - 1,
    }
    contact_frames: dict[int, np.ndarray] = {}
    frames_with_boxes = 0

    for frame_index in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        row = rows_by_frame.get(frame_index, {})
        for det in row.get("detections", []):
            box = det.get("bbox_xyxy", [])
            if len(box) != 4:
                continue
            x1, y1, _, _ = draw_box(frame, box, GREEN, 1)
            tid = det.get("track_id")
            label = f"{det.get('class_name', 'object')} id={tid}" if tid is not None else str(det.get("class_name", "object"))
            cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)

        active_boxes = 0
        for item_id, values in series.items():
            box = values[frame_index]
            x1, y1, _, _ = draw_box(frame, box, RED, max(3, width // 500))
            exact = frame_index in source_points[item_id]
            prefix = "ACCIDENT ACTOR" if args.domain == "synthetic" else "SUSPECT ACCIDENT VEHICLE"
            suffix = "" if exact else " (interpolated)"
            draw_label(frame, f"{prefix} {item_id}{suffix}", x1, y1 - 6, RED)
            active_boxes += 1
        if active_boxes:
            frames_with_boxes += 1

        rx1, ry1, _, _ = draw_box(frame, region, ORANGE, max(2, width // 900))
        cv2.putText(frame, "DATASET ACCIDENT REGION", (rx1, max(20, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2, cv2.LINE_AA)

        banner_h = max(78, int(round(height * 0.105)))
        overlay = frame.copy()
        banner_color = (15, 35, 150) if detected else (20, 115, 180)
        cv2.rectangle(overlay, (0, 0), (width, banner_h), banner_color, -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
        decision = "ACCIDENT DETECTED" if detected else "ACCIDENT MISSED"
        cv2.putText(frame, f"{args.domain.upper()} TEST | VideoMAE: {decision} | score={score:.4f} threshold={threshold:.2f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, max(0.65, width / 1800.0), WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Ground truth: ACCIDENT ({metadata.get('type', 'unknown')}) | t={frame_index / fps:.2f}s | event={event_time:.2f}s", (18, 65), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, width / 2200.0), WHITE, 2, cv2.LINE_AA)
        footer = "RED: CARLA collision actor ground truth" if args.domain == "synthetic" else "RED: suspected accident track linked to annotated collision region"
        cv2.putText(frame, footer, (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 2300.0), RED, 2, cv2.LINE_AA)

        writer.write(frame)
        if frame_index in capture_indices:
            contact_frames[frame_index] = frame.copy()

    cap.release()
    writer.release()
    make_contact_sheet(contact_frames, Path(args.output_contact_sheet))

    evidence = {
        "domain": args.domain,
        "video": args.video,
        "video_frames": total_frames,
        "fps": fps,
        "event_time_sec": event_time,
        "ground_truth": "accident",
        "accident_type": metadata.get("type"),
        "videomae_accident_score": score,
        "threshold": threshold,
        "model_detected_accident": detected,
        "evidence_method": evidence_method,
        "selected_accident_vehicle_ids": selected_ids,
        "association_scores": association_scores,
        "source_box_frames": {str(item_id): len(source_points[item_id]) for item_id in selected_ids},
        "rendered_frame_box_coverage": frames_with_boxes / max(total_frames, 1),
        "interpolation_note": "Missing boxes inside/outside observed ranges are linearly interpolated or nearest-filled for report continuity.",
        "visualization_boundary": "Boxes identify dataset collision actors (synthetic) or suspected tracks associated with the annotated collision region (real); they are not VideoMAE box predictions.",
        "output_video": str(output_video),
        "contact_sheet": args.output_contact_sheet,
    }
    Path(args.output_evidence_json).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
