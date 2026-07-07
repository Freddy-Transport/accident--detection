#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import ensure_dir, load_data_config, read_metadata_rows, video_metadata, write_csv

FIELDS = [
    'sample_id', 'source_video_id', 'video_path', 'source_dataset', 'original_label',
    'mapped_label', 'label_id', 'label_type', 'window_type', 'duration', 'fps',
    'num_frames', 'width', 'height', 'split', 'scene_id', 'camera_id', 'group_id',
    'start_frame', 'end_frame', 'clip_start_sec', 'clip_end_sec', 'notes'
]


def sec(frame: int, fps: float) -> float:
    return round(frame / fps, 6) if fps > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    meta_rows = read_metadata_rows(cfg['metadata_real'])
    if args.limit:
        meta_rows = meta_rows[:args.limit]

    video_root = Path(cfg['video_root'])
    manifest = []
    corrupt = []
    ext = Counter()
    types = Counter()
    splits = Counter()
    resolutions = Counter()

    for row in meta_rows:
        rel = row['path']
        path = video_root / rel
        source_video_id = Path(rel).stem
        ext[path.suffix.lower()] += 1
        types[row.get('type', 'unknown')] += 1
        split = row.get('split_in_distribution') or row.get('split_geo_aware') or 'unknown'
        splits[split] += 1

        meta = video_metadata(path)
        if not meta['opened']:
            corrupt.append(str(path))
        else:
            resolutions[f"{meta['width']}x{meta['height']}"] += 1

        fps = float(meta['fps'] or 0.0)
        frames = int(meta['num_frames'] or row.get('no_frames') or 0)
        duration = float(meta['duration'] or row.get('duration') or 0.0)
        accident_time = float(row.get('accident_time') or 0.0)
        accident_frame = int(round(accident_time * fps)) if fps > 0 else int(float(row.get('accident_frame') or 0))
        acc_start = max(0, accident_frame - int(round(float(cfg['accident_clip']['pre_sec']) * fps))) if fps > 0 else 0
        acc_end = min(frames, accident_frame + int(round(float(cfg['accident_clip']['post_sec']) * fps))) if fps > 0 and frames else frames

        base = {
            'source_video_id': source_video_id,
            'video_path': str(path),
            'source_dataset': 'ACCIDENT-real',
            'original_label': row.get('type', 'accident'),
            'duration': round(duration, 6),
            'fps': round(fps, 6),
            'num_frames': frames,
            'width': meta['width'],
            'height': meta['height'],
            'split': split if split in {'train', 'test'} else 'unknown',
            'scene_id': row.get('region') or row.get('scene_layout') or '',
            'camera_id': f"{row.get('region', 'unknown')}::{row.get('scene_layout', 'unknown')}",
            'group_id': source_video_id,
        }

        manifest.append({
            **base,
            'sample_id': f'{source_video_id}__accident',
            'mapped_label': 'accident',
            'label_id': 1,
            'label_type': 'weak_clip_event_window',
            'window_type': 'accident_window',
            'start_frame': acc_start,
            'end_frame': acc_end,
            'clip_start_sec': sec(acc_start, fps),
            'clip_end_sec': sec(acc_end, fps),
            'notes': f"accident_time={accident_time};type={row.get('type')};weak_clip=true",
        })

        neg_end_sec = max(0.0, accident_time - float(cfg['non_accident_clip']['margin_before_accident_sec']))
        if neg_end_sec >= float(cfg['non_accident_clip']['min_duration_sec']) and fps > 0:
            neg_start = int(round(float(cfg['non_accident_clip']['start_sec']) * fps))
            neg_end = min(frames, int(round(neg_end_sec * fps)))
            if neg_end > neg_start:
                manifest.append({
                    **base,
                    'sample_id': f'{source_video_id}__pre_event_non_accident',
                    'mapped_label': 'non_accident',
                    'label_id': 0,
                    'label_type': 'weak_clip_pre_event',
                    'window_type': 'pre_event_non_accident',
                    'start_frame': neg_start,
                    'end_frame': neg_end,
                    'clip_start_sec': sec(neg_start, fps),
                    'clip_end_sec': sec(neg_end, fps),
                    'notes': f'derived_pre_accident_until={neg_end_sec:.3f};weak_negative=true',
                })

    write_csv(cfg['manifest_path'], manifest, FIELDS)
    label_map = Path(cfg['label_map_path'])
    ensure_dir(label_map.parent)
    label_map.write_text(json.dumps({'non_accident': 0, 'accident': 1, 'hard_negative': 2, 'uncertain': 3}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    Path(cfg['class_names_path']).write_text('non_accident\naccident\n', encoding='utf-8')

    audit = {
        'dataset_root': cfg['accident_dataset_root'],
        'metadata_rows': len(meta_rows),
        'manifest_rows': len(manifest),
        'video_extensions': dict(ext),
        'original_type_counts': dict(types),
        'split_counts': dict(splits),
        'resolution_top10': resolutions.most_common(10),
        'corrupt_or_unreadable_count': len(corrupt),
        'corrupt_or_unreadable_examples': corrupt[:20],
        'has_explicit_non_accident': False,
        'has_explicit_hard_negative': False,
        'label_granularity': ['video-level accident type', 'event-level accident_time', 'bbox-level accident region metadata'],
        'weak_negative_policy': 'pre-event clips before accident_time minus margin',
        'manifest_fields': FIELDS,
    }
    reports = ensure_dir(Path(cfg['experiment_root']) / 'reports')
    (reports / 'dataset_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    md = [
        '# ACCIDENT Dataset Audit',
        '',
        f"- Dataset root: `{cfg['accident_dataset_root']}`",
        f"- Metadata rows: `{len(meta_rows)}`",
        f"- Manifest rows: `{len(manifest)}`",
        f"- Video extensions: `{dict(ext)}`",
        f"- Accident type counts: `{dict(types)}`",
        f"- Split counts: `{dict(splits)}`",
        '- Explicit non-accident labels: `false`',
        '- Explicit hard-negative labels: `false`',
        '- Manifest is clip-level and grouped by `source_video_id` / `group_id`.',
        '',
        'ACCIDENT real is accident-centric. Non-accident samples in this baseline are weak pre-event clips, not real normal traffic videos.',
    ]
    (reports / 'dataset_audit_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
