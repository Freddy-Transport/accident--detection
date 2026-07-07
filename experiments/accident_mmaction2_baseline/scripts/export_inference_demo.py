#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import torch
from mmaction.apis import inference_recognizer, init_recognizer

from common import ensure_dir, load_experiment_config, timestamp

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
ANNOTATION_DIR = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/data/annotations'


def read_annotations(path: Path, limit: int = 0) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        video_path, label = line.rsplit(maxsplit=1)
        rows.append((video_path, int(label)))
        if limit and len(rows) >= limit:
            break
    return rows


def video_meta(video_path: str) -> dict[str, float | int]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        'fps': fps,
        'num_frames': frames,
        'width': width,
        'height': height,
        'duration': float(frames / fps) if fps > 0 else 0.0,
        'start_frame': 0,
        'end_frame': max(0, frames - 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='videomae', choices=['videomae', 'slowfast', 'x3d'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    cfg = load_experiment_config()
    model_cfg = cfg['models'][args.model]['config']
    output_dir = Path(args.output_dir or Path(cfg['outputs_root']) / f'{timestamp()}_{args.model}_inference_demo')
    pred_dir = ensure_dir(output_dir / 'predictions')
    report_dir = ensure_dir(output_dir / 'reports')

    ann_file = ANNOTATION_DIR / f'{args.split}.txt'
    annotations = read_annotations(ann_file, args.limit)
    started = time.time()
    model = init_recognizer(model_cfg, args.checkpoint, device=args.device)

    rows = []
    latencies = []
    for video_path, label_id in annotations:
        item_started = time.perf_counter()
        result = inference_recognizer(model, video_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - item_started) * 1000.0
        latencies.append(latency_ms)
        scores = result.pred_score.detach().cpu().float().numpy().tolist()
        pred_label = int(result.pred_label.detach().cpu().numpy().reshape(-1)[0])
        meta = video_meta(video_path)
        rows.append({
            'sample_id': Path(video_path).stem,
            'video_path': video_path,
            'split': args.split,
            'label_id': label_id,
            'pred_label': pred_label,
            'prob_non_accident': float(scores[0]) if len(scores) > 0 else 0.0,
            'prob_accident': float(scores[1]) if len(scores) > 1 else 0.0,
            'latency_ms': latency_ms,
            **meta,
        })

    csv_path = pred_dir / 'predictions.csv'
    json_path = pred_dir / 'predictions.json'
    fieldnames = list(rows[0].keys()) if rows else [
        'sample_id', 'video_path', 'split', 'label_id', 'pred_label',
        'prob_non_accident', 'prob_accident', 'latency_ms', 'fps', 'num_frames',
        'width', 'height', 'duration', 'start_frame', 'end_frame'
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    summary = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'config': model_cfg,
        'split': args.split,
        'num_samples': len(rows),
        'elapsed_sec': round(time.time() - started, 3),
        'avg_latency_ms': round(sum(latencies) / len(latencies), 3) if latencies else None,
        'fps_clips': round(1000.0 / (sum(latencies) / len(latencies)), 3) if latencies else None,
        'predictions_csv': str(csv_path),
        'predictions_json': str(json_path),
    }
    (report_dir / 'inference_demo_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
