#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from common import ensure_dir, load_data_config, read_manifest


def scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / float(max(width, height))
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    return new_w, new_h


def resize_to_max_side(frame, max_side: int):
    height, width = frame.shape[:2]
    new_w, new_h = scaled_size(width, height, max_side)
    if (new_w, new_h) == (width, height):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clip_video_cv2(src: str, dst: Path, fps: float, start_frame: int, end_frame: int, max_side: int, overwrite: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return False
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_width <= 0 or src_height <= 0:
        cap.release()
        return False
    out_width, out_height = scaled_size(src_width, src_height, max_side)
    out_fps = fps if fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    ensure_dir(dst.parent)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_width, out_height))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_idx = max(0, start_frame)
    ok_count = 0
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(resize_to_max_side(frame, max_side))
        ok_count += 1
        frame_idx += 1
    writer.release()
    cap.release()
    return ok_count > 0 and dst.exists() and dst.stat().st_size > 0


def clip_video(src: str, dst: Path, fps: float, start_frame: int, end_frame: int, max_side: int, overwrite: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return True
    ensure_dir(dst.parent)
    start = max(0.0, start_frame / max(fps, 1e-6))
    dur = max(0.1, (end_frame - start_frame) / max(fps, 1e-6))
    scale_filter = []
    if max_side > 0:
        scale_filter = ['-vf', f'scale=if(gt(iw,ih),{max_side},-2):if(gt(iw,ih),-2,{max_side})']
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-ss', f'{start:.3f}', '-i', src,
        '-t', f'{dur:.3f}', *scale_filter, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-an', str(dst)
    ]
    try:
        if subprocess.run(cmd, check=False).returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return True
    except FileNotFoundError:
        pass
    return clip_video_cv2(src, dst, fps, start_frame, end_frame, max_side=max_side, overwrite=overwrite)


def grouped_train_val(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    groups = sorted({r.get('group_id') or r.get('source_video_id') or r['sample_id'] for r in rows if r.get('split') == 'train'})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_fraction))) if groups else 0
    return set(groups[n_val:]), set(groups[:n_val])


def select_balanced(rows: list[dict[str, Any]], max_per_label: int, rng: random.Random) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row['label_id'])].append(row)
    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        selected.extend(items if max_per_label == 0 else items[:max_per_label])
    selected.sort(key=lambda r: (r.get('group_id') or r.get('source_video_id') or r['sample_id'], int(r['label_id'])))
    return selected


def profile_settings(cfg: dict[str, Any], profile: str | None, max_per_split: int | None) -> dict[str, Any]:
    if not profile:
        per_label = max_per_split if max_per_split is not None else 20
        return {
            'name': 'smoke',
            'annotation_dir': Path(cfg['annotation_dir']),
            'clip_root': Path(cfg['clip_root']),
            'max_per_label': {'train': per_label, 'val': per_label, 'test': per_label},
            'materialize': True,
            'max_side': int(cfg.get('clip_materialization', {}).get('max_side', 640)),
            'overwrite': bool(cfg.get('clip_materialization', {}).get('overwrite', False)),
        }
    profiles = cfg.get('annotation_profiles', {})
    if profile not in profiles:
        raise KeyError(f'Unknown profile `{profile}`. Available profiles: {sorted(profiles)}')
    p = profiles[profile]
    subdir = p.get('annotation_subdir') or profile
    ann_dir = Path(cfg.get('annotation_profiles_root', cfg['annotation_dir'])) / subdir if subdir != '.' else Path(cfg['annotation_dir'])
    clip_root = Path(cfg.get('generated_clip_root') or cfg['clip_root']) / (p.get('clip_subdir') or profile)
    return {
        'name': profile,
        'annotation_dir': ann_dir,
        'clip_root': clip_root,
        'max_per_label': p.get('max_per_label', {'train': 0, 'val': 0, 'test': 0}),
        'materialize': bool(p.get('materialize', True)),
        'max_side': int(p.get('max_side', cfg.get('clip_materialization', {}).get('max_side', 640))),
        'overwrite': bool(p.get('overwrite', cfg.get('clip_materialization', {}).get('overwrite', False))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--profile', choices=['smoke', 'small', 'medium', 'full'])
    parser.add_argument('--max-per-split', type=int, default=None, help='Legacy per-label limit for smoke/default output.')
    parser.add_argument('--no-materialize', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    rows = read_manifest(cfg['manifest_path'])
    train_groups, val_groups = grouped_train_val(rows, float(cfg['val_fraction_from_train']), int(cfg['random_seed']))

    buckets = {'train': [], 'val': [], 'test': []}
    for row in rows:
        group = row.get('group_id') or row.get('source_video_id') or row['sample_id']
        if row.get('split') == 'test':
            buckets['test'].append(row)
        elif group in val_groups:
            buckets['val'].append(row)
        elif group in train_groups:
            buckets['train'].append(row)

    settings = profile_settings(cfg, args.profile, args.max_per_split)
    materialize = settings['materialize'] and not args.no_materialize
    overwrite = args.overwrite or settings['overwrite']
    ann_dir = settings['annotation_dir']
    clip_root = settings['clip_root']
    ensure_dir(ann_dir)
    ensure_dir(clip_root)

    link_path = Path(cfg['experiment_root']) / 'data' / 'generated_clips'
    if cfg.get('generated_clip_root') and not link_path.exists():
        try:
            os.symlink(Path(cfg['generated_clip_root']), link_path, target_is_directory=True)
        except FileExistsError:
            pass

    rng = random.Random(int(cfg['random_seed']))
    summary: dict[str, Any] = {
        'profile': settings['name'],
        'annotation_dir': str(ann_dir),
        'clip_root': str(clip_root),
        'materialize': materialize,
        'max_side': settings['max_side'],
        'splits': {},
    }
    used_groups: dict[str, set[str]] = {}
    exit_code = 0

    for split, split_rows in buckets.items():
        limit = int(settings['max_per_label'].get(split, 0))
        selected = select_balanced(split_rows, limit, rng)
        ann_path = ann_dir / f'{split}.txt'
        clip_csv = ann_dir / f'{split}_clips.csv'
        written = []
        failed = []
        with ann_path.open('w', encoding='utf-8') as ann_file:
            for row in selected:
                clip_path = clip_root / split / f"{row['sample_id']}.mp4"
                ok = True
                if materialize:
                    ok = clip_video(
                        row['video_path'], clip_path, float(row['fps']), int(row['start_frame']), int(row['end_frame']),
                        max_side=int(settings['max_side']), overwrite=overwrite,
                    )
                if not ok:
                    failed.append(row['sample_id'])
                    continue
                ann_file.write(f"{clip_path} {row['label_id']}\n")
                written.append({**row, 'clip_path': str(clip_path), 'annotation_split': split, 'annotation_profile': settings['name']})
        if written:
            fields = list(written[0].keys())
            with clip_csv.open('w', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=fields)
                writer.writeheader()
                writer.writerows(written)
        groups = {r.get('group_id') or r.get('source_video_id') or r['sample_id'] for r in written}
        used_groups[split] = groups
        summary['splits'][split] = {
            'available': len(split_rows),
            'selected': len(selected),
            'written': len(written),
            'failed': failed[:20],
            'labels': dict(Counter(r['mapped_label'] for r in written)),
            'ann_file': str(ann_path),
            'clip_csv': str(clip_csv),
        }
        if failed:
            exit_code = 1

    leakage = sorted((used_groups.get('train', set()) & used_groups.get('val', set())) | (used_groups.get('train', set()) & used_groups.get('test', set())) | (used_groups.get('val', set()) & used_groups.get('test', set())))
    summary['group_leakage_count'] = len(leakage)
    summary['group_leakage_examples'] = leakage[:20]
    if leakage:
        exit_code = 1

    (ann_dir / 'annotation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    reports = ensure_dir(Path(cfg['experiment_root']) / 'reports')
    (reports / f'annotation_summary_{settings["name"]}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
