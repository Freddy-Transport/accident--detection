#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_accident_rnd.model import run_smoke_training


def current_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train R3D-18 accident baseline.")
    parser.add_argument("--smoke", action="store_true", help="Run a single synthetic batch and save a checkpoint.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "checkpoints" / "smoke_r3d18.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("Only --smoke training is implemented in the MVP. Use manifest training after real data is added.")
    metadata = run_smoke_training(args.output, device=args.device, pretrained=args.pretrained)
    metadata.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": current_commit(), "checkpoint_path": str(args.output)})
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
