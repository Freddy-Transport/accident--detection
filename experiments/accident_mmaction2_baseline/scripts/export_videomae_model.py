#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path('/root/autodl-tmp/traffic_accident_rnd')
DEFAULT_CONFIG = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/configs/videomae_pretrained_full_accident.py'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/outputs/20260707_144326_videomae_pretrained_full_3epoch/best_acc_top1_epoch_2.pth'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'models/exported/mmaction2/videomae_full'
DEFAULT_REPORT = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/reports/model_export_report.md'
DEFAULT_TEST_ANN = PROJECT_ROOT / 'experiments/accident_mmaction2_baseline/data/annotations/full/test.txt'
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


class VideoMAELogitsWrapper(torch.nn.Module):
    def __init__(self, recognizer: torch.nn.Module):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise RuntimeError('frames must have shape [B,3,16,224,224]')
        x = frames.unsqueeze(1)
        feat = self.recognizer.extract_feat(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return self.recognizer.cls_head(feat)


def find_sample_video(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == '1':
            return ' '.join(parts[:-1])
    return None


def load_clip_tensor(video: str | None, *, clip_len: int = 16, size: int = 224) -> torch.Tensor:
    if not video:
        return torch.randn(1, 3, clip_len, size, size, dtype=torch.float32)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return torch.randn(1, 3, clip_len, size, size, dtype=torch.float32)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return torch.randn(1, 3, clip_len, size, size, dtype=torch.float32)
    indices = np.linspace(0, max(total - 1, 0), clip_len).round().astype(int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        short = min(h, w)
        if short <= 0:
            continue
        scale = 256.0 / short
        resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
        h2, w2 = resized.shape[:2]
        y0 = max(0, (h2 - size) // 2)
        x0 = max(0, (w2 - size) // 2)
        crop = resized[y0:y0 + size, x0:x0 + size]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size))
        crop = (crop.astype(np.float32) - MEAN) / STD
        frames.append(crop)
    cap.release()
    if len(frames) < clip_len:
        return torch.randn(1, 3, clip_len, size, size, dtype=torch.float32)
    arr = np.stack(frames, axis=0)  # T,H,W,C
    arr = np.transpose(arr, (3, 0, 1, 2))  # C,T,H,W
    return torch.from_numpy(arr).unsqueeze(0).float()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item())


def write_reports(report_md: Path, report_json: Path, payload: dict[str, Any]) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# VideoMAE Export Report',
        '',
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Config: `{payload['config']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Input shape: `{payload['input_shape']}`",
        f"- TorchScript: `{payload['torchscript']['status']}` `{payload['torchscript'].get('path', '')}` diff `{payload['torchscript'].get('max_abs_diff')}`",
        f"- ONNX: `{payload['onnx']['status']}` `{payload['onnx'].get('path', '')}` diff `{payload['onnx'].get('max_abs_diff')}`",
        '',
        '## Notes',
        '',
        '- Exported logits expect preprocessed RGB tensors with MMAction2 mean/std normalization.',
        '- Keep MMAction2 config and preprocessing with the exported model; `.pt`/`.onnx` alone is not the full video pipeline.',
    ]
    if payload['onnx'].get('reason'):
        lines += ['', '## ONNX Blocker', '', payload['onnx']['reason']]
    report_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Export fine-tuned VideoMAE logits wrapper to TorchScript and ONNX.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--report-md', default=str(DEFAULT_REPORT))
    parser.add_argument('--sample-video', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from mmaction.apis import init_recognizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md = Path(args.report_md)
    report_json = output_dir / 'model_export_report.json'
    ts_path = output_dir / 'videomae_full.ts.pt'
    onnx_path = output_dir / 'videomae_full.onnx'
    sample_video = args.sample_video or find_sample_video(DEFAULT_TEST_ANN)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    recognizer = init_recognizer(args.config, args.checkpoint, device=str(device)).eval()
    wrapper = VideoMAELogitsWrapper(recognizer).eval().to(device)
    sample = load_clip_tensor(sample_video).to(device)
    with torch.no_grad():
        reference_logits = wrapper(sample)

    torchscript = {'status': 'blocked'}
    try:
        traced = torch.jit.trace(wrapper, sample, check_trace=False)
        traced.save(str(ts_path))
        loaded = torch.jit.load(str(ts_path), map_location=device).eval()
        with torch.no_grad():
            ts_logits = loaded(sample)
        torchscript = {
            'status': 'ok',
            'path': str(ts_path),
            'max_abs_diff': max_abs_diff(reference_logits, ts_logits),
            'size_bytes': ts_path.stat().st_size,
        }
    except Exception as exc:
        torchscript = {'status': 'failed', 'reason': repr(exc)}

    onnx = {'status': 'blocked'}
    try:
        torch.onnx.export(
            wrapper,
            sample,
            str(onnx_path),
            input_names=['frames'],
            output_names=['logits'],
            opset_version=args.opset,
            do_constant_folding=True,
        )
        onnx = {'status': 'exported', 'path': str(onnx_path), 'size_bytes': onnx_path.stat().st_size, 'opset': args.opset}
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            ort_logits = session.run(['logits'], {'frames': sample.detach().cpu().numpy()})[0]
            onnx['status'] = 'ok'
            onnx['max_abs_diff'] = float(np.max(np.abs(reference_logits.detach().cpu().numpy() - ort_logits)))
        except Exception as exc:
            onnx['status'] = 'exported_unverified'
            onnx['reason'] = f'ONNXRuntime verification failed: {exc!r}'
    except Exception as exc:
        if args.opset != 18:
            try:
                torch.onnx.export(wrapper, sample, str(onnx_path), input_names=['frames'], output_names=['logits'], opset_version=18, do_constant_folding=True)
                onnx = {'status': 'exported', 'path': str(onnx_path), 'size_bytes': onnx_path.stat().st_size, 'opset': 18, 'reason': f'opset {args.opset} failed, opset 18 exported without runtime verification: {exc!r}'}
            except Exception as exc2:
                onnx = {'status': 'failed', 'reason': f'opset {args.opset} failed: {exc!r}; opset 18 failed: {exc2!r}'}
        else:
            onnx = {'status': 'failed', 'reason': repr(exc)}

    payload = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'sample_video': sample_video,
        'input_shape': list(sample.shape),
        'reference_logits': reference_logits.detach().cpu().float().numpy().tolist(),
        'torchscript': torchscript,
        'onnx': onnx,
    }
    write_reports(report_md, report_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if torchscript.get('status') == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
