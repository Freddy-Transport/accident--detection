#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
LEGACY_ROOT = PROJECT_ROOT / 'third_party/highway_inference_legacy'
for path in [SRC_ROOT, LEGACY_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import numpy as np

from traffic_accident_rnd.cascade import LEGACY_TRAJECTORY_REASON_MAP, read_jsonl, write_jsonl

ID_LIST_NAMES = [
    'weiting_ids',
    'trajectory_anomaly_ids',
    'warning_sign_ids',
    'low_speed_ids',
    'legacy_accident_ids',
    'congestion_ids',
]


def _video_shape(video: str | None, frames: list[dict[str, Any]]) -> tuple[int, int]:
    if video:
        cap = cv2.VideoCapture(video)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if width > 0 and height > 0:
                return height, width
        cap.release()
    max_x = max((float(v) for frame in frames for det in frame.get('detections', []) for v in det.get('bbox_xyxy', [0, 0, 0, 0])[0::2]), default=1920.0)
    max_y = max((float(v) for frame in frames for det in frame.get('detections', []) for v in det.get('bbox_xyxy', [0, 0, 0, 0])[1::2]), default=1080.0)
    return max(1, int(max_y) + 10), max(1, int(max_x) + 10)


def _legacy_sequence(frames: list[dict[str, Any]]) -> list[np.ndarray]:
    sequence: list[np.ndarray] = []
    for frame in sorted(frames, key=lambda item: int(item.get('frame_index', 0))):
        rows = []
        for det in frame.get('detections', []):
            if 'track_id' not in det:
                continue
            bbox = det.get('bbox_xyxy', [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            rows.append([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1), float(det['track_id'])])
        sequence.append(np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64))
    return sequence


def _events(flags: list[int], id_lists: dict[str, list[int]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, (reason, key) in LEGACY_TRAJECTORY_REASON_MAP.items():
        ids = sorted({int(v) for v in id_lists.get(key, [])})
        flag = int(flags[idx]) if idx < len(flags) else 0
        if flag != 1 and not ids:
            continue
        events.append({
            'reason': reason,
            'legacy_flag_index': idx,
            'legacy_flag_value': flag,
            'legacy_event_type': key,
            'evidence_track_ids': ids,
            'candidate_score': 1.0 + 0.25 * len(ids) + float(flag),
            'send_to_video_model': True,
        })
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description='Run legacy trackVehicleInSeqpre trajectory event logic on Track JSONL.')
    parser.add_argument('--tracks-jsonl', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--video', default=None)
    parser.add_argument('--video-id', default=None)
    parser.add_argument('--roi-mask', default=None)
    parser.add_argument('--legacy-log', default=None)
    args = parser.parse_args()

    try:
        from trackVehicleInSeq_pre0907 import trackVehicleInSeqpre
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'reason': f'legacy trajectory import failed: {exc}'}, ensure_ascii=False, indent=2))
        return 2

    frames = read_jsonl(args.tracks_jsonl)
    height, width = _video_shape(args.video, frames)
    if args.roi_mask:
        segmentation_map = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if segmentation_map is None:
            print(json.dumps({'status': 'blocked', 'reason': f'could not read roi mask: {args.roi_mask}'}, ensure_ascii=False, indent=2))
            return 2
    else:
        segmentation_map = np.full((height, width), 255, dtype=np.uint8)
    sequence = _legacy_sequence(frames)
    log_handle = None
    if args.legacy_log:
        log_path = Path(args.legacy_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8')
    try:
        if log_handle:
            with contextlib.redirect_stdout(log_handle):
                result = trackVehicleInSeqpre(sequence, segmentation_map)
        else:
            result = trackVehicleInSeqpre(sequence, segmentation_map)
    finally:
        if log_handle:
            log_handle.close()
    flags = [int(v) for v in result[0]]
    id_lists = {name: sorted({int(v) for v in values}) for name, values in zip(ID_LIST_NAMES, result[1:])}
    events = _events(flags, id_lists)
    video_id = args.video_id or Path(args.tracks_jsonl).stem
    payload = {
        'video_id': video_id,
        'tracks_jsonl': args.tracks_jsonl,
        'legacy_flags': flags,
        'legacy_id_lists': id_lists,
        'events': events,
        'notes': 'Legacy trajectory events are candidate evidence only; final accident classification is VideoMAE score gated.',
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_jsonl(events, args.output_jsonl)
    print(json.dumps({'video_id': video_id, 'events': len(events), 'output_json': str(output_json), 'output_jsonl': args.output_jsonl}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
