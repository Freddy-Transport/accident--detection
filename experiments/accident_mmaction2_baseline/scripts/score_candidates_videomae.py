#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import sys

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
from traffic_accident_rnd.cascade import read_jsonl


def extract_segment(video: str, output: Path, start_sec: float, end_sec: float) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = max(start_frame + 1, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    written = 0
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        idx += 1
    writer.release()
    cap.release()
    return written > 0 and output.exists() and output.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Score candidate segments with a fine-tuned VideoMAE checkpoint.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--candidates-jsonl', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    try:
        import torch
        from mmaction.apis import inference_recognizer, init_recognizer
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'reason': f'MMAction2 import failed: {exc}'}, ensure_ascii=False, indent=2))
        return 2

    output_dir = Path(args.output_dir)
    clip_dir = output_dir / 'candidate_clips'
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(args.candidates_jsonl)
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    rows = []
    for idx, candidate in enumerate(candidates):
        start = float(candidate['segment_start_sec'])
        end = float(candidate['segment_end_sec'])
        clip_path = clip_dir / f'candidate_{idx:03d}_{start:.2f}_{end:.2f}.mp4'
        if not extract_segment(args.video, clip_path, start, end):
            rows.append({**candidate, 'candidate_id': idx, 'clip_path': str(clip_path), 'status': 'decode_failed', 'accident_score': 0.0, 'pred_label': 0})
            continue
        t0 = time.perf_counter()
        result = inference_recognizer(model, str(clip_path))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        scores = result.pred_score.detach().cpu().float().numpy().tolist()
        accident_score = float(scores[1]) if len(scores) > 1 else 0.0
        rows.append({
            **candidate,
            'candidate_id': idx,
            'clip_path': str(clip_path),
            'status': 'ok',
            'prob_non_accident': float(scores[0]) if scores else 0.0,
            'accident_score': accident_score,
            'pred_label': 1 if accident_score >= args.threshold else 0,
            'threshold': args.threshold,
            'latency_ms': (time.perf_counter() - t0) * 1000.0,
        })
    json_path = pred_dir / 'candidate_predictions.json'
    csv_path = pred_dir / 'candidate_predictions.csv'
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = sorted({key for row in rows for key in row.keys()}) if rows else ['candidate_id', 'accident_score', 'pred_label']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'video': args.video, 'candidates': len(candidates), 'predictions_json': str(json_path), 'predictions_csv': str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
