#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
PIPELINE_SCRIPT = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/scripts/run_event_pipeline.py'
DEFAULT_VIDEO_ROOT = Path(os.environ.get('TRAFFIC_USER_VIDEO_ROOT', '/autodl-fs/data/traffic_accident_rnd/user_videos'))
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def is_stream(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))


def find_latest_video(video_root: Path) -> Path:
    candidates = [p for p in video_root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(f'no video files found under {video_root}')
    return max(candidates, key=lambda item: item.stat().st_mtime)


def resolve_video(value: str | None, *, video_root: Path, latest: bool) -> str:
    if latest:
        return str(find_latest_video(video_root))
    if not value:
        raise ValueError('provide --video <path/name> or --latest')
    if is_stream(value):
        return value
    raw = Path(value).expanduser()
    if raw.is_absolute() and raw.exists():
        return str(raw)
    rooted = video_root / value
    if rooted.exists():
        return str(rooted)
    if raw.exists():
        return str(raw.resolve())
    raise FileNotFoundError(f'video not found: {value}; checked absolute path, cwd, and {video_root}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the full accident event pipeline for a newly uploaded video.')
    parser.add_argument('--video', default=None, help='Absolute video path, RTSP/HTTP URL, or filename relative to --video-root.')
    parser.add_argument('--latest', action='store_true', help='Run the newest video under --video-root.')
    parser.add_argument('--video-root', default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--threshold', type=float, default=0.56)
    parser.add_argument('--tracker', choices=['auto', 'strongsort', 'iou'], default='auto')
    parser.add_argument('--frame-stride', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--push-endpoint', default=os.environ.get('TRAFFIC_EVENT_PUSH_ENDPOINT'))
    args = parser.parse_args()

    video_root = Path(args.video_root)
    video_root.mkdir(parents=True, exist_ok=True)
    video = resolve_video(args.video, video_root=video_root, latest=args.latest)

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        '--video', video,
        '--threshold', str(args.threshold),
        '--tracker', args.tracker,
        '--frame-stride', str(args.frame_stride),
        '--device', args.device,
    ]
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    if args.push:
        cmd.append('--push')
    if args.push_endpoint:
        cmd.extend(['--push-endpoint', args.push_endpoint])

    print(json.dumps({'video': video, 'video_root': str(video_root), 'cmd': cmd}, ensure_ascii=False, indent=2))
    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == '__main__':
    raise SystemExit(main())
