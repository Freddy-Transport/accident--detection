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

import cv2

from traffic_accident_rnd.cascade import build_accident_evidence_tracks, read_jsonl


def positive_windows(predictions: list[dict], threshold: float) -> list[dict]:
    return [p for p in predictions if float(p.get('accident_score', 0.0)) >= threshold]


def main() -> int:
    parser = argparse.ArgumentParser(description='Render accident-positive candidate windows with evidence boxes.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--tracks-jsonl', required=True)
    parser.add_argument('--predictions-json', required=True)
    parser.add_argument('--output-video', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--evidence-json', default=None)
    args = parser.parse_args()

    tracks = read_jsonl(args.tracks_jsonl)
    predictions = json.loads(Path(args.predictions_json).read_text(encoding='utf-8'))
    windows = positive_windows(predictions, args.threshold)
    evidence_rows = build_accident_evidence_tracks(predictions, tracks, threshold=args.threshold)
    evidence_json = Path(args.evidence_json) if args.evidence_json else Path(args.output_video).with_name('accident_evidence_tracks.json')
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text(json.dumps(evidence_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    by_frame = {int(frame.get('frame_index', 0)): frame.get('detections', []) for frame in tracks}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(json.dumps({'status': 'blocked', 'reason': f'could not open video: {args.video}'}, ensure_ascii=False, indent=2))
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output = Path(args.output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(json.dumps({'status': 'blocked', 'reason': f'could not open writer: {output}'}, ensure_ascii=False, indent=2))
        return 2

    frame_index = 0
    rendered_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_index / fps
        active = [w for w in windows if float(w['segment_start_sec']) <= ts <= float(w['segment_end_sec'])]
        if active:
            track_ids = {int(tid) for w in active for tid in w.get('evidence_track_ids', [])}
            score = max(float(w.get('accident_score', 0.0)) for w in active)
            reasons = sorted({reason for w in active for reason in w.get('trigger_reasons', [])})
            banner = f"ACCIDENT DETECTED score={score:.3f} reasons={','.join(reasons)[:72]}"
            cv2.rectangle(frame, (0, 0), (width, 68), (0, 0, 180), -1)
            cv2.putText(frame, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, 'Red boxes are SUSPECT accident evidence vehicles, not box-level ground truth', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for det in by_frame.get(frame_index, []):
                det_track_id = int(det.get('track_id', -1))
                if track_ids and det_track_id not in track_ids:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in det.get('bbox_xyxy', [0, 0, 0, 0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                label = f"SUSPECT_ACCIDENT_VEHICLE id={det_track_id} {det.get('class_name','')}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                ly = max(72, y1 - 8)
                cv2.rectangle(frame, (x1, max(0, ly - th - 8)), (min(width - 1, x1 + tw + 8), ly + 4), (0, 0, 180), -1)
                cv2.putText(frame, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            rendered_frames += 1
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    print(json.dumps({'video': args.video, 'output_video': str(output), 'evidence_json': str(evidence_json), 'evidence_rows': len(evidence_rows), 'positive_windows': len(windows), 'rendered_frames': rendered_frames}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
