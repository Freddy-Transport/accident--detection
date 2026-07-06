#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic remote smoke video.")
    parser.add_argument("--output", type=Path, default=Path("data/samples/smoke_accident_like.mp4"))
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 320, 180
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))
    total_frames = int(args.fps * args.seconds)
    for frame_idx in range(total_frames):
        t = frame_idx / args.fps
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        cv2.line(frame, (0, 130), (width, 130), (90, 90, 90), 2)
        car1_x = int(20 + min(t, 2.6) * 45)
        car2_x = int(260 - min(t, 2.6) * 42)
        if 2.4 <= t <= 3.4:
            shake = int((frame_idx % 2) * 12)
            car1_x += shake
            car2_x -= shake
            cv2.circle(frame, (160, 96), 22 + shake, (0, 180, 255), 3)
            cv2.line(frame, (130, 72), (190, 120), (0, 220, 255), 2)
        cv2.rectangle(frame, (car1_x, 92), (car1_x + 46, 120), (40, 170, 255), -1)
        cv2.rectangle(frame, (car2_x, 102), (car2_x + 44, 128), (220, 80, 60), -1)
        writer.write(frame)
    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
