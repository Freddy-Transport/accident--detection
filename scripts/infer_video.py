#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_accident_rnd.inference import predict_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Run accident candidate detection and baseline scoring on a video.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inference" / "prediction.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--track-path", type=Path, default=None)
    parser.add_argument("--threshold-z", type=float, default=1.5)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = predict_video(
        args.video,
        checkpoint_path=args.checkpoint,
        track_path=args.track_path,
        output_path=args.output,
        threshold_z=args.threshold_z,
        sample_fps=args.sample_fps,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
