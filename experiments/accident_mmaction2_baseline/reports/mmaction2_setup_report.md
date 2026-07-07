# MMAction2 Setup Report

- Host: `autodl-container-4f124886f3-4146060b`
- Project: `/root/autodl-tmp/traffic_accident_rnd`
- GPU: NVIDIA GeForce RTX 4080 SUPER 32GB, driver 595.71.05.
- Isolated env: `/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction`
- Verified versions: `torch 2.8.0+cu128`, `mmcv 2.1.0`, `mmengine 0.10.7`, `mmaction 1.2.0`, `decord 0.6.0`.
- `ffmpeg` command is missing; clip extraction uses OpenCV fallback.
- `mmcv-lite>=2.0.0rc4,<2.2.0`, `setuptools>=68`, `importlib_metadata` are pinned/installed for Python 3.12 + MIM compatibility.
- `patch_mmaction_package.py` removes a missing `drn` import from the isolated pip package; report: `patch_mmaction_package_report.json`.
- GitHub source clone of `open-mmlab/mmaction2` failed from the remote due network timeout; MIM uses installed package `.mim/tools`.

Validation commands:

```bash
experiments/accident_mmaction2_baseline/scripts/setup_mmaction_env.sh
.venv_mmaction/bin/python -c "import torch, mmcv, mmengine, mmaction, decord, cv2, av"
.venv_mmaction/bin/mim train --help
```

Base commit before this stage commit: `989f427`.
