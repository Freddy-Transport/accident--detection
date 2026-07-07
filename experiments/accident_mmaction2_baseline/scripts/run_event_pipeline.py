#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_accident_rnd.cascade import build_final_accident_events, read_jsonl

SCRIPTS = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/scripts'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth'
DEFAULT_CONFIG = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py'
DEFAULT_OUTPUTS = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/outputs'


def sanitize_video_id(video: str) -> str:
    stem = Path(video).stem if '://' not in video else video.rsplit('/', 1)[-1]
    stem = stem or 'stream'
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)[:80]


def run_step(name: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{name}.log'
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finished = time.time()
    payload = {'cmd': cmd, 'returncode': proc.returncode, 'started_at': started, 'finished_at': finished, 'elapsed_sec': finished - started, 'output': proc.stdout}
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'step': name, 'returncode': proc.returncode, 'log': str(log_path)}, ensure_ascii=False))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def maybe_push_events(final_payload: dict[str, Any], output_dir: Path, *, push: bool, endpoint: str | None) -> dict[str, Any]:
    enabled = push and os.environ.get('TRAFFIC_EVENT_PUSH_ENABLED') == '1' and endpoint
    if not enabled:
        result = {'mode': 'dry_run', 'reason': 'set --push, TRAFFIC_EVENT_PUSH_ENABLED=1 and endpoint to enable real push', 'events': final_payload.get('events', [])}
        write_json(output_dir / 'event_push_dry_run.json', result)
        return result
    import requests
    response = requests.post(endpoint, json=final_payload, timeout=10)
    result = {'mode': 'pushed', 'endpoint': endpoint, 'status_code': response.status_code, 'response_text': response.text[:1000]}
    write_json(output_dir / 'event_push_result.json', result)
    response.raise_for_status()
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description='Run YOLO -> StrongSORT/IoU -> trajectory -> candidates -> VideoMAE accident event pipeline.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--threshold', type=float, default=0.56)
    parser.add_argument('--tracker', choices=['auto', 'strongsort', 'iou'], default='auto')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--yolo-model', default=str(PROJECT_ROOT / 'models/pretrained/车辆检测_v8l.pt'))
    parser.add_argument('--yolo-python', default=str(PROJECT_ROOT / 'third_party/ACCIDENT/baselines/heuristic/.venv/bin/python'))
    parser.add_argument('--mmaction-python', default=str(PROJECT_ROOT / '.venv_mmaction/bin/python'))
    parser.add_argument('--pipeline-python', default=sys.executable)
    parser.add_argument('--pre-window-sec', type=float, default=2.0)
    parser.add_argument('--post-window-sec', type=float, default=4.0)
    parser.add_argument('--max-segments', type=int, default=8)
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--push-endpoint', default=os.environ.get('TRAFFIC_EVENT_PUSH_ENDPOINT'))
    args = parser.parse_args()

    video_id = sanitize_video_id(args.video)
    timestamp = time.strftime('%Y%m%d_%H%M')
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUTS / f'{timestamp}_event_pipeline_{video_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    detections = output_dir / 'detections.jsonl'
    tracks = output_dir / 'tracks.jsonl'
    trajectory_json = output_dir / 'trajectory_events.json'
    trajectory_jsonl = output_dir / 'trajectory_events.jsonl'
    candidates = output_dir / 'candidate_segments.jsonl'
    visualization = output_dir / 'visualization.mp4'

    write_json(output_dir / 'pipeline_config.json', {
        'video': args.video,
        'video_id': video_id,
        'checkpoint': args.checkpoint,
        'config': args.config,
        'threshold': args.threshold,
        'tracker': args.tracker,
        'frame_stride': args.frame_stride,
        'yolo_model': args.yolo_model,
        'pre_window_sec': args.pre_window_sec,
        'post_window_sec': args.post_window_sec,
        'max_segments': args.max_segments,
        'push_requested': bool(args.push),
        'push_endpoint_configured': bool(args.push_endpoint),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    })

    run_step('01_yolo_detect', [args.yolo_python, str(SCRIPTS / 'run_yolo_detect.py'), '--video', args.video, '--model', args.yolo_model, '--output-jsonl', str(detections), '--frame-stride', str(args.frame_stride)], log_dir)
    run_step('02_track', [args.pipeline_python, str(SCRIPTS / 'run_strongsort_tracker.py'), '--video', args.video, '--detections-jsonl', str(detections), '--output-jsonl', str(tracks), '--tracker', args.tracker], log_dir)
    run_step('03_legacy_trajectory', [args.pipeline_python, str(SCRIPTS / 'run_legacy_trajectory.py'), '--video', args.video, '--tracks-jsonl', str(tracks), '--output-json', str(trajectory_json), '--output-jsonl', str(trajectory_jsonl), '--video-id', video_id, '--legacy-log', str(log_dir / 'legacy_trackVehicleInSeqpre_stdout.log')], log_dir)
    run_step('04_trigger_candidates', [args.pipeline_python, str(SCRIPTS / 'trigger_candidates_from_tracks.py'), '--tracks-jsonl', str(tracks), '--trajectory-events-json', str(trajectory_json), '--output-jsonl', str(candidates), '--video-id', video_id, '--pre-window-sec', str(args.pre_window_sec), '--post-window-sec', str(args.post_window_sec), '--max-segments', str(args.max_segments)], log_dir)

    candidate_rows = read_jsonl(candidates)
    if candidate_rows:
        run_step('05_videomae_score', [args.mmaction_python, str(SCRIPTS / 'score_candidates_videomae.py'), '--video', args.video, '--candidates-jsonl', str(candidates), '--config', args.config, '--checkpoint', args.checkpoint, '--output-dir', str(output_dir), '--threshold', str(args.threshold)], log_dir)
        predictions_json = output_dir / 'predictions/candidate_predictions.json'
        predictions = json.loads(predictions_json.read_text(encoding='utf-8'))
        shutil.copy2(predictions_json, output_dir / 'videomae_predictions.json')
    else:
        predictions = []
        write_json(output_dir / 'videomae_predictions.json', predictions)
        write_json(output_dir / 'predictions/candidate_predictions.json', predictions)

    final_events = build_final_accident_events(predictions, video_id=video_id, video_path=args.video, threshold=args.threshold, checkpoint=args.checkpoint)
    final_payload = {
        'video': args.video,
        'video_id': video_id,
        'threshold': args.threshold,
        'checkpoint': args.checkpoint,
        'candidate_count': len(candidate_rows),
        'event_count': len(final_events),
        'events': final_events,
        'notes': 'Final accident events require VideoMAE accident_score >= threshold; YOLO/Track evidence is auxiliary.',
    }
    write_json(output_dir / 'final_events.json', final_payload)
    maybe_push_events(final_payload, output_dir, push=args.push, endpoint=args.push_endpoint)

    predictions_for_render = output_dir / 'predictions/candidate_predictions.json'
    run_step('06_render', [args.pipeline_python, str(SCRIPTS / 'render_accident_video.py'), '--video', args.video, '--tracks-jsonl', str(tracks), '--predictions-json', str(predictions_for_render), '--output-video', str(visualization), '--threshold', str(args.threshold)], log_dir)

    summary = {
        **final_payload,
        'output_dir': str(output_dir),
        'detections_jsonl': str(detections),
        'tracks_jsonl': str(tracks),
        'trajectory_events_json': str(trajectory_json),
        'candidate_segments_jsonl': str(candidates),
        'videomae_predictions_json': str(output_dir / 'videomae_predictions.json'),
        'final_events_json': str(output_dir / 'final_events.json'),
        'visualization_video': str(visualization),
    }
    write_json(output_dir / 'event_pipeline_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
