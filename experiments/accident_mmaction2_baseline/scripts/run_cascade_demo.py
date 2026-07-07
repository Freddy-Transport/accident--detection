#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(json.dumps({'cmd': cmd}, ensure_ascii=False))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run YOLO -> IoU track -> candidate trigger -> VideoMAE -> render cascade demo.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='/root/autodl-tmp/traffic_accident_rnd/experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--yolo-model', default='/root/autodl-tmp/traffic_accident_rnd/models/pretrained/车辆检测_v8l.pt')
    parser.add_argument('--yolo-python', default='/root/autodl-tmp/traffic_accident_rnd/third_party/ACCIDENT/baselines/heuristic/.venv/bin/python')
    parser.add_argument('--mmaction-python', default='/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python')
    parser.add_argument('--frame-stride', type=int, default=5)
    args = parser.parse_args()

    root = Path('/root/autodl-tmp/traffic_accident_rnd')
    scripts = root / 'experiments/accident_mmaction2_baseline/scripts'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    detections = out / f'{stem}_detections.jsonl'
    tracks = out / f'{stem}_tracks.jsonl'
    candidates = out / f'{stem}_candidate_segments.jsonl'
    render = out / f'{stem}_accident_visualization.mp4'

    run([args.yolo_python, str(scripts / 'run_yolo_detect.py'), '--video', args.video, '--model', args.yolo_model, '--output-jsonl', str(detections), '--frame-stride', str(args.frame_stride)])
    run([sys.executable, str(scripts / 'run_iou_tracker.py'), '--detections-jsonl', str(detections), '--output-jsonl', str(tracks)])
    run([sys.executable, str(scripts / 'trigger_candidates_from_tracks.py'), '--tracks-jsonl', str(tracks), '--output-jsonl', str(candidates), '--video-id', stem])
    run([args.mmaction_python, str(scripts / 'score_candidates_videomae.py'), '--video', args.video, '--candidates-jsonl', str(candidates), '--config', args.config, '--checkpoint', args.checkpoint, '--output-dir', str(out), '--threshold', str(args.threshold)])
    predictions = out / 'predictions/candidate_predictions.json'
    run([sys.executable, str(scripts / 'render_accident_video.py'), '--video', args.video, '--tracks-jsonl', str(tracks), '--predictions-json', str(predictions), '--output-video', str(render), '--threshold', str(args.threshold)])
    summary = {'video': args.video, 'output_dir': str(out), 'detections_jsonl': str(detections), 'tracks_jsonl': str(tracks), 'candidates_jsonl': str(candidates), 'predictions_json': str(predictions), 'visualization_video': str(render)}
    (out / 'cascade_demo_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
