#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_accident_rnd.cascade import build_track_candidate_segments, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description='Build accident candidate segments from track JSONL.')
    parser.add_argument('--tracks-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--video-id', default=None)
    parser.add_argument('--pre-window-sec', type=float, default=2.0)
    parser.add_argument('--post-window-sec', type=float, default=4.0)
    parser.add_argument('--max-segments', type=int, default=8)
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
    write_jsonl(segments, args.output_jsonl)
    print(json.dumps({'tracks_jsonl': args.tracks_jsonl, 'output_jsonl': args.output_jsonl, 'segments': len(segments)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
