#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_accident_rnd.schemas import validate_manifest_file, validate_track_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manifest or YOLO/Track JSONL files.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=["manifest", "track"], default="manifest")
    args = parser.parse_args()
    errors = validate_track_file(args.path) if args.kind == "track" else validate_manifest_file(args.path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"ok: {args.kind} schema valid: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
