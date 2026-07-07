#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from common import ensure_dir, load_experiment_config

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
MIM_BIN = PROJECT_ROOT / '.venv_mmaction/bin/mim'

MODEL_SPECS = {
    'videomae': {
        'config_id': 'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400',
        'canonical': 'videomae_base_k400_backbone.pth',
        'description': 'VideoMAE ViT-Base Kinetics-400 pretrained backbone',
    },
    'slowfast': {
        'config_id': 'slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb',
        'canonical': 'slowfast_r50_k400_backbone.pth',
        'description': 'SlowFast R50 Kinetics-400 pretrained backbone',
    },
    'x3d': {
        'config_id': 'x3d_s_13x6x1_facebook-kinetics400-rgb',
        'canonical': 'x3d_s_k400_backbone.pth',
        'description': 'X3D-S Kinetics-400 pretrained backbone',
    },
}


def pth_files(path: Path) -> set[Path]:
    return {p.resolve() for p in path.glob('*.pth') if p.is_file()}


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else 0.0


def strip_head(source: Path, target: Path) -> dict[str, Any]:
    ensure_dir(target.parent)
    checkpoint = torch.load(source, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    stripped = {k: v for k, v in state.items() if not k.startswith('cls_head.') and '.cls_head.' not in k}
    removed = sorted(set(state) - set(stripped))
    out = {
        'state_dict': stripped,
        'meta': {
            'source_checkpoint': str(source),
            'processed_by': 'prepare_pretrained_checkpoints.py',
            'head_keys_removed': len(removed),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }
    torch.save(out, target)
    return {'target': str(target), 'target_size_mb': file_size_mb(target), 'removed_head_keys': removed[:20], 'removed_head_key_count': len(removed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['videomae'], choices=sorted(MODEL_SPECS))
    parser.add_argument('--dest')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ignore-ssl', action='store_true')
    args = parser.parse_args()

    cfg = load_experiment_config()
    dest = ensure_dir(Path(args.dest or cfg.get('pretrained_dir') or PROJECT_ROOT / 'models/pretrained/mmaction2'))
    raw_dir = ensure_dir(dest / 'raw_mim_downloads')
    report: dict[str, Any] = {'dest': str(dest), 'raw_dir': str(raw_dir), 'dry_run': args.dry_run, 'models': {}}
    exit_code = 0

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        target = dest / spec['canonical']
        item: dict[str, Any] = {'description': spec['description'], 'config_id': spec['config_id'], 'target': str(target)}
        if target.exists() and target.stat().st_size > 0:
            item.update({'status': 'exists', 'target_size_mb': file_size_mb(target)})
            report['models'][model_name] = item
            continue
        if args.dry_run:
            item.update({'status': 'dry_run', 'command': [str(MIM_BIN), 'download', 'mmaction2', '--config', spec['config_id'], '--dest', str(raw_dir)]})
            report['models'][model_name] = item
            continue
        if not MIM_BIN.exists():
            item.update({'status': 'blocked', 'reason': f'Missing MIM binary: {MIM_BIN}'})
            report['models'][model_name] = item
            exit_code = 2
            continue
        before = pth_files(raw_dir)
        cmd = [str(MIM_BIN), 'download', 'mmaction2', '--config', spec['config_id'], '--dest', str(raw_dir)]
        if args.ignore_ssl:
            cmd.append('--ignore-ssl')
        env = os.environ.copy()
        env.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        after = pth_files(raw_dir)
        candidates = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
        item['download_command'] = cmd
        item['download_returncode'] = proc.returncode
        item['download_log_tail'] = proc.stdout[-4000:]
        if proc.returncode != 0 or not candidates:
            item.update({'status': 'blocked', 'reason': 'mim download failed or produced no .pth checkpoint'})
            report['models'][model_name] = item
            exit_code = 2
            continue
        source = candidates[0]
        item.update({'status': 'downloaded', 'source_checkpoint': str(source), 'source_size_mb': file_size_mb(source)})
        try:
            item.update(strip_head(source, target))
            item['status'] = 'ready'
        except Exception as exc:
            item.update({'status': 'blocked', 'reason': f'failed to strip classifier head: {exc}'})
            exit_code = 2
        report['models'][model_name] = item

    reports_dir = ensure_dir(Path(cfg['reports_root']))
    json_path = reports_dir / 'pretrained_checkpoint_report.json'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Pretrained Checkpoint Report', '']
    for name, item in report['models'].items():
        lines.extend([
            f"## {name}",
            f"- Status: `{item.get('status')}`",
            f"- Config id: `{item.get('config_id')}`",
            f"- Target: `{item.get('target')}`",
            f"- Target size MB: `{item.get('target_size_mb', 'n/a')}`",
            '',
        ])
        if item.get('reason'):
            lines.extend([f"- Blocked reason: `{item['reason']}`", ''])
    (reports_dir / 'pretrained_checkpoint_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
