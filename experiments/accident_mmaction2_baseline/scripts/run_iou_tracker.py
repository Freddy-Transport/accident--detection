#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_accident_rnd.cascade import assign_iou_tracks, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description='Assign lightweight IoU tracks to detection JSONL.')
    parser.add_argument('--detections-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.3)
    parser.add_argument('--max-age-frames', type=int, default=8)
    args = parser.parse_args()

    frames = read_jsonl(args.detections_jsonl)
    tracked = assign_iou_tracks(frames, iou_threshold=args.iou_threshold, max_age_frames=args.max_age_frames)
    write_jsonl(tracked, args.output_jsonl)
    track_ids = sorted({det['track_id'] for frame in tracked for det in frame.get('detections', []) if 'track_id' in det})
    print(json.dumps({'detections_jsonl': args.detections_jsonl, 'output_jsonl': args.output_jsonl, 'frames': len(tracked), 'tracks': len(track_ids)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
