#!/usr/bin/env python3
"""Evaluate completed annotation-free inference against withheld labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2

from traffic_accident_rnd.cascade import iou_xyxy, read_jsonl


def _video_size(video: str, metadata: dict[str, Any]) -> tuple[int, int]:
    cap = cv2.VideoCapture(video)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width > 0 and height > 0:
            return width, height
    cap.release()
    return int(metadata.get('width', 0)), int(metadata.get('height', 0))


def _region_pixels(metadata: dict[str, Any], width: int, height: int) -> list[float]:
    return [
        float(metadata['x1']) * width, float(metadata['y1']) * height,
        float(metadata['x2']) * width, float(metadata['y2']) * height,
    ]


def _center_inside(bbox: list[float], region: list[float]) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate final events after annotation-free inference has completed.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--final-events-json', required=True)
    parser.add_argument('--tracks-jsonl', required=True)
    parser.add_argument('--metadata-json', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--spatial-time-tolerance-sec', type=float, default=1.0)
    args = parser.parse_args()

    final_payload = json.loads(Path(args.final_events_json).read_text(encoding='utf-8'))
    if final_payload.get('annotation_free_inference') is not True:
        raise RuntimeError('Refusing evaluation: inference output is not marked annotation_free_inference=true')
    if final_payload.get('annotation_inputs'):
        raise RuntimeError('Refusing evaluation: inference declared non-empty annotation_inputs')
    metadata = json.loads(Path(args.metadata_json).read_text(encoding='utf-8'))
    tracks = read_jsonl(args.tracks_jsonl)
    events = list(final_payload.get('events', []))
    ground_truth_time = float(metadata['accident_time'])
    width, height = _video_size(args.video, metadata)
    region = _region_pixels(metadata, width, height)

    first_event = min(events, key=lambda event: float(event.get('detected_time_sec', 0.0)), default=None)
    best_event = min(events, key=lambda event: abs(
        float(event.get('detected_time_sec', 0.0)) - ground_truth_time), default=None)
    evidence_ids = {int(value) for value in (best_event or {}).get('evidence_track_ids', [])}
    spatial_rows: list[dict[str, Any]] = []
    for frame in tracks:
        timestamp = float(frame.get('timestamp_sec', 0.0))
        if abs(timestamp - ground_truth_time) > args.spatial_time_tolerance_sec:
            continue
        for detection in frame.get('detections', []):
            track_id = detection.get('track_id')
            bbox = [float(value) for value in detection.get('bbox_xyxy', [])]
            if track_id is None or int(track_id) not in evidence_ids or len(bbox) != 4:
                continue
            overlap = iou_xyxy(bbox, region)
            spatial_rows.append({
                'frame_index': int(frame.get('frame_index', 0)),
                'timestamp_sec': timestamp,
                'track_id': int(track_id),
                'bbox_xyxy': bbox,
                'region_iou': overlap,
                'center_inside_accident_region': _center_inside(bbox, region),
            })

    best_spatial = max(
        spatial_rows,
        key=lambda row: (float(row['region_iou']), bool(row['center_inside_accident_region'])),
        default=None,
    )
    detected_time = float(best_event['detected_time_sec']) if best_event else None
    first_alarm_time = float(first_event['detected_time_sec']) if first_event else None
    payload = {
        'evaluation_only': True,
        'annotation_used_during_inference': False,
        'metadata_loaded_after_inference': True,
        'video': args.video,
        'ground_truth': {
            'accident_time_sec': ground_truth_time,
            'accident_type': metadata.get('type'),
            'accident_region_normalized': [metadata['x1'], metadata['y1'], metadata['x2'], metadata['y2']],
        },
        'prediction': {
            'accident_detected': best_event is not None,
            'detected_time_sec': detected_time,
            'temporal_error_sec': abs(detected_time - ground_truth_time) if detected_time is not None else None,
            'candidate_contains_ground_truth_time': (
                float(best_event['segment_start_sec']) <= ground_truth_time <= float(best_event['segment_end_sec'])
                if best_event else False
            ),
            'accident_score': float(best_event['accident_score']) if best_event else None,
            'trigger_reasons': list(best_event.get('trigger_reasons', [])) if best_event else [],
            'suspect_track_ids': sorted(evidence_ids),
        },
        'alarm_summary': {
            'event_count': len(events),
            'first_alarm_time_sec': first_alarm_time,
            'first_alarm_offset_sec': (
                first_alarm_time - ground_truth_time if first_alarm_time is not None else None
            ),
            'matched_event_id': best_event.get('event_id') if best_event else None,
            'matched_event_is_first_alarm': bool(best_event and first_event and best_event == first_event),
        },
        'spatial_evaluation': {
            'matched': bool(best_spatial and (best_spatial['region_iou'] >= 0.1
                                             or best_spatial['center_inside_accident_region'])),
            'best_evidence': best_spatial,
            'evaluated_track_boxes': len(spatial_rows),
        },
        'all_predicted_events': events,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
