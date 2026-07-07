#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import json
from pathlib import Path

import mmaction


def main() -> int:
    package_root = Path(mmaction.__file__).parent
    localizers = package_root / 'models' / 'localizers'
    init_file = localizers / '__init__.py'
    drn_dir = localizers / 'drn'
    result = {
        'package_root': str(package_root),
        'patched': False,
        'reason': '',
        'target': str(init_file),
    }
    if not init_file.exists():
        result['reason'] = 'localizers __init__.py missing'
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    if drn_dir.exists():
        result['reason'] = 'drn module exists; no patch needed'
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    text = init_file.read_text(encoding='utf-8')
    patched = text.replace('from .drn.drn import DRN\n', '')
    patched = patched.replace("__all__ = ['TEM', 'PEM', 'BMN', 'TCANet', 'DRN']", "__all__ = ['TEM', 'PEM', 'BMN', 'TCANet']")
    if patched != text:
        init_file.write_text(patched, encoding='utf-8')
        result['patched'] = True
        result['reason'] = 'removed missing drn import from pip package localizers init'
    else:
        result['reason'] = 'no matching drn import found'
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
