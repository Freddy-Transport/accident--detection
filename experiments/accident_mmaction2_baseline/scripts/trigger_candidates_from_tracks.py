#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_accident_rnd.cascade import build_legacy_candidate_segments, build_track_candidate_segments, merge_candidate_segments, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description='Build accident candidate segments from track JSONL.')
    parser.add_argument('--tracks-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--video-id', default=None)
    parser.add_argument('--pre-window-sec', type=float, default=2.0)
    parser.add_argument('--post-window-sec', type=float, default=4.0)
    parser.add_argument('--max-segments', type=int, default=8)
    parser.add_argument('--trajectory-events-json', default=None)
    args = parser.parse_args()

    frames = read_jsonl(args.tracks_jsonl)
    video_id = args.video_id or Path(args.tracks_jsonl).stem
    segments = build_track_candidate_segments(
        frames,
        video_id=video_id,
        pre_window_sec=args.pre_window_sec,
        post_window_sec=args.post_window_sec,
        max_segments=args.max_segments,
    )
    legacy_segments = []
    if args.trajectory_events_json:
        trajectory_events = json.loads(Path(args.trajectory_events_json).read_text(encoding='utf-8'))
        legacy_segments = build_legacy_candidate_segments(
            frames,
            trajectory_events,
            video_id=video_id,
            pre_window_sec=args.pre_window_sec,
            post_window_sec=args.post_window_sec,
            max_segments=args.max_segments,
        )
    segments = merge_candidate_segments(segments + legacy_segments)
    segments.sort(key=lambda item: float(item.get('candidate_score', 0.0)), reverse=True)
    segments = sorted(segments[:args.max_segments], key=lambda item: float(item['segment_start_sec']))
    write_jsonl(segments, args.output_jsonl)
    print(json.dumps({'tracks_jsonl': args.tracks_jsonl, 'trajectory_events_json': args.trajectory_events_json, 'output_jsonl': args.output_jsonl, 'track_segments': len(segments) - len(legacy_segments), 'legacy_segments': len(legacy_segments), 'segments': len(segments)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
