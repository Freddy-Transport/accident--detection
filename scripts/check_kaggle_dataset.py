#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/traffic_accident_rnd")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "dataset"


def parse_size_to_gb(value: str) -> float | None:
    text = value.strip().upper().replace(" ", "")
    units = [("TB", 1024.0), ("GB", 1.0), ("MB", 1 / 1024.0), ("KB", 1 / (1024.0 * 1024.0)), ("B", 1 / (1024.0**3))]
    for suffix, factor in units:
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * factor
            except ValueError:
                return None
    try:
        return float(text) / (1024.0**3)
    except ValueError:
        return None


def write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Kaggle dataset availability and size before download.")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, for example owner/dataset")
    parser.add_argument("--max-gb", type=float, default=5.0)
    parser.add_argument("--download", action="store_true", help="Download only when total size is known and within --max-gb")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or DEFAULT_OUTPUT_DIR / f"kaggle_check_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    token_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", "/root/.kaggle")) / "kaggle.json"
    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "max_gb": args.max_gb,
        "download_requested": args.download,
        "token_path": str(token_path),
    }

    if not token_path.exists():
        status.update({"status": "blocked_no_token", "message": "Kaggle token is missing; no download attempted."})
        write_status(output, status)
        return 0
    if shutil.which("kaggle") is None:
        status.update({"status": "blocked_no_kaggle_cli", "message": "kaggle CLI not found in PATH."})
        write_status(output, status)
        return 0

    cmd = ["kaggle", "datasets", "files", "-d", args.dataset, "--csv"]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    status["list_command"] = " ".join(cmd)
    status["list_returncode"] = result.returncode
    if result.returncode != 0:
        status.update({"status": "blocked_list_failed", "stderr": result.stderr[-2000:]})
        write_status(output, status)
        return 0

    rows = list(csv.DictReader(result.stdout.splitlines()))
    total_gb = 0.0
    unknown_sizes = []
    files = []
    for row in rows:
        size_text = row.get("size") or row.get("Size") or ""
        size_gb = parse_size_to_gb(size_text)
        files.append({"name": row.get("name") or row.get("Name"), "size": size_text, "size_gb": size_gb})
        if size_gb is None:
            unknown_sizes.append(row)
        else:
            total_gb += size_gb
    status.update({"status": "listed", "file_count": len(files), "estimated_total_gb": total_gb, "unknown_size_count": len(unknown_sizes), "files": files[:100]})

    if unknown_sizes or total_gb > args.max_gb:
        status["download_allowed"] = False
        status["message"] = "Download blocked because size is unknown or exceeds limit."
        write_status(output, status)
        return 0
    status["download_allowed"] = True
    if args.download:
        target = PROJECT_ROOT / "data" / "downloads" / args.dataset.replace("/", "__")
        target.mkdir(parents=True, exist_ok=True)
        dl_cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(target), "--unzip"]
        dl_result = subprocess.run(dl_cmd, check=False, text=True, capture_output=True)
        status.update({"download_command": " ".join(dl_cmd), "download_returncode": dl_result.returncode, "download_dir": str(target)})
        if dl_result.returncode != 0:
            status.update({"status": "download_failed", "stderr": dl_result.stderr[-2000:]})
            write_status(output, status)
            return 1
        status["status"] = "downloaded"
    write_status(output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
