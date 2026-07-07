#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ensure_dir, load_experiment_config, timestamp

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
MMACTION_ROOT = PROJECT_ROOT / 'third_party' / 'mmaction2'
MIM_BIN = PROJECT_ROOT / '.venv_mmaction' / 'bin' / 'mim'


def has_mmaction() -> tuple[bool, str]:
    try:
        import mmaction  # noqa: F401
        import mmcv  # noqa: F401
        import mmengine  # noqa: F401
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)


def resolve_runner(tool: str) -> tuple[str, Path] | None:
    tool_path = MMACTION_ROOT / 'tools' / tool
    if tool_path.exists():
        return 'source', tool_path
    if MIM_BIN.exists():
        return 'mim', MIM_BIN
    return None


def build_cmd(mode: str, config_path: str, work_dir: Path, checkpoint: str | None) -> list[str]:
    tool = 'train.py' if mode == 'train' else 'test.py'
    runner = resolve_runner(tool)
    if runner is None:
        raise FileNotFoundError(f'Missing MMAction2 tool and MIM binary: {tool}')
    runner_kind, runner_path = runner
    if runner_kind == 'source':
        if mode == 'train':
            return [sys.executable, str(runner_path), config_path, '--work-dir', str(work_dir)]
        if not checkpoint:
            raise ValueError('test mode requires --checkpoint')
        return [sys.executable, str(runner_path), config_path, checkpoint, '--work-dir', str(work_dir)]
    if mode == 'train':
        return [str(runner_path), 'train', 'mmaction2', config_path, '--work-dir', str(work_dir), '--gpus', '1']
    if not checkpoint:
        raise ValueError('test mode requires --checkpoint')
    return [str(runner_path), 'test', 'mmaction2', config_path, '--checkpoint', checkpoint, '--work-dir', str(work_dir), '--gpus', '1']


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='videomae_pretrained')
    parser.add_argument('--config')
    parser.add_argument('--work-dir')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    cfg = load_experiment_config()
    if args.model not in cfg['models'] and not args.config:
        print(json.dumps({'status': 'blocked', 'reason': f"Unknown model `{args.model}`. Available: {sorted(cfg['models'])}"}, ensure_ascii=False, indent=2))
        return 2
    ok, msg = has_mmaction()
    if not ok:
        print(json.dumps({'status': 'blocked', 'reason': 'MMAction2 import failed: ' + msg}, ensure_ascii=False, indent=2))
        return 2

    config_path = args.config or cfg['models'][args.model]['config']
    work_dir = Path(args.work_dir or Path(cfg['outputs_root']) / f'{timestamp()}_{args.model}')
    ensure_dir(work_dir)
    try:
        cmd = build_cmd('train', config_path, work_dir, args.checkpoint)
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc)}, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps({'status': 'running', 'mode': 'train', 'model': args.model, 'config': config_path, 'work_dir': str(work_dir), 'cmd': cmd}, ensure_ascii=False, indent=2))
    env = os.environ.copy()
    env.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    return subprocess.run(cmd, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
