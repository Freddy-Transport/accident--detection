#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
LEGACY_ROOT = PROJECT_ROOT / 'third_party/highway_inference_legacy'
for path in [SRC_ROOT, LEGACY_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import numpy as np

from traffic_accident_rnd.cascade import assign_iou_tracks, iou_xyxy, read_jsonl, write_jsonl


def _class_index(name: str, mapping: dict[str, int]) -> int:
    if name not in mapping:
        mapping[name] = len(mapping)
    return mapping[name]


def _detections_to_np(detections: list[dict[str, Any]], class_to_idx: dict[str, int]) -> np.ndarray:
    rows = []
    for det in detections:
        bbox = det.get('bbox_xyxy', [])
        if len(bbox) != 4:
            continue
        rows.append([
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
            float(det.get('confidence', 0.0)),
            float(_class_index(str(det.get('class_name', 'object')), class_to_idx)),
        ])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 6), dtype=np.float64)


def _best_detection(track_bbox_xyxy: list[float], detections: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_iou = 0.0
    for det in detections:
        score = iou_xyxy(track_bbox_xyxy, det.get('bbox_xyxy', []))
        if score > best_iou:
            best_iou = score
            best = det
    return {**best, 'matched_iou': best_iou} if best else {'matched_iou': 0.0}


def _run_iou_fallback(frames: list[dict[str, Any]], output_jsonl: str, reason: str) -> dict[str, Any]:
    tracked = assign_iou_tracks(frames)
    write_jsonl(tracked, output_jsonl)
    return {'tracker': 'iou_fallback', 'fallback_reason': reason, 'frames': len(tracked), 'output_jsonl': output_jsonl}


def main() -> int:
    parser = argparse.ArgumentParser(description='Run legacy StrongSORT tracker on YOLO detection JSONL with IoU fallback.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--detections-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--tracker', choices=['auto', 'strongsort', 'iou'], default='auto')
    parser.add_argument('--track-thresh', type=float, default=0.5)
    parser.add_argument('--match-thresh', type=float, default=0.8)
    parser.add_argument('--track-buffer', type=int, default=30)
    args = parser.parse_args()

    frames = read_jsonl(args.detections_jsonl)
    if args.tracker == 'iou':
        print(json.dumps(_run_iou_fallback(frames, args.output_jsonl, 'forced_iou'), ensure_ascii=False, indent=2))
        return 0

    try:
        from strongsort_tracker import StrongSORTTracker
    except Exception as exc:
        if args.tracker == 'strongsort':
            print(json.dumps({'status': 'blocked', 'reason': f'StrongSORT import failed: {exc}'}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(_run_iou_fallback(frames, args.output_jsonl, f'StrongSORT import failed: {exc}'), ensure_ascii=False, indent=2))
        return 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        if args.tracker == 'strongsort':
            print(json.dumps({'status': 'blocked', 'reason': f'could not open video: {args.video}'}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(_run_iou_fallback(frames, args.output_jsonl, f'could not open video: {args.video}'), ensure_ascii=False, indent=2))
        return 0

    try:
        tracker_args = SimpleNamespace(track_thresh=args.track_thresh, match_thresh=args.match_thresh, track_buffer=args.track_buffer)
        tracker = StrongSORTTracker(tracker_args, frame_rate=30)
        by_index = {int(frame.get('frame_index', 0)): frame for frame in frames}
        target_indices = set(by_index)
        max_index = max(target_indices) if target_indices else -1
        class_to_idx: dict[str, int] = {}
        prev: dict[int, tuple[float, float, float]] = {}
        out_rows: list[dict[str, Any]] = []
        frame_index = 0
        while frame_index <= max_index:
            ok, image = cap.read()
            if not ok:
                break
            if frame_index not in target_indices:
                frame_index += 1
                continue
            src = by_index[frame_index]
            detections = list(src.get('detections', []))
            det_np = _detections_to_np(detections, class_to_idx)
            tracks = tracker.update(det_np, image)
            out_dets = []
            timestamp = float(src.get('timestamp_sec', 0.0))
            for item in tracks:
                x, y, w, h, tid = [float(v) for v in item]
                bbox = [x, y, x + w, y + h]
                matched = _best_detection(bbox, detections)
                cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                tid_int = int(tid)
                if tid_int in prev and timestamp > prev[tid_int][2]:
                    px, py, pts = prev[tid_int]
                    speed = float(np.hypot(cx - px, cy - py) / max(timestamp - pts, 1e-6))
                else:
                    speed = 0.0
                prev[tid_int] = (cx, cy, timestamp)
                out_dets.append({
                    'track_id': tid_int,
                    'class_name': str(matched.get('class_name', 'vehicle')),
                    'confidence': float(matched.get('confidence', 0.0)),
                    'bbox_xyxy': [round(float(v), 3) for v in bbox],
                    'center_xy': [round(cx, 3), round(cy, 3)],
                    'speed_px_per_sec': round(speed, 6),
                    'matched_iou': round(float(matched.get('matched_iou', 0.0)), 6),
                })
            out_rows.append({**src, 'detections': out_dets})
            frame_index += 1
    except Exception as exc:
        cap.release()
        if args.tracker == 'strongsort':
            print(json.dumps({'status': 'blocked', 'reason': f'StrongSORT runtime failed: {exc}'}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(_run_iou_fallback(frames, args.output_jsonl, f'StrongSORT runtime failed: {exc}'), ensure_ascii=False, indent=2))
        return 0
    cap.release()

    write_jsonl(out_rows, args.output_jsonl)
    print(json.dumps({'tracker': 'strongsort', 'frames': len(out_rows), 'output_jsonl': args.output_jsonl, 'classes': class_to_idx}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
