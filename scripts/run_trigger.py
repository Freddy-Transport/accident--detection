#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_accident_rnd.trigger import detect_candidate_segments, write_candidate_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frame-difference candidate trigger on a video.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/candidates/candidate_segments.jsonl"))
    parser.add_argument("--threshold-z", type=float, default=2.5)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    args = parser.parse_args()
    segments = detect_candidate_segments(args.video, threshold_z=args.threshold_z, sample_fps=args.sample_fps)
    write_candidate_segments(segments, args.output)
    print(f"wrote {len(segments)} candidate segment(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
