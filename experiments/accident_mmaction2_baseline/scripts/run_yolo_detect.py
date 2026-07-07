#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from traffic_accident_rnd.cascade import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description='Run YOLO vehicle/person detection and export JSONL frames.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--model', default='/root/autodl-tmp/traffic_accident_rnd/models/pretrained/车辆检测_v8l.pt')
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--frame-stride', type=int, default=5)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--classes', default='car,truck,bus,motorcycle,bicycle,person')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'reason': f'ultralytics import failed: {exc}'}, ensure_ascii=False, indent=2))
        return 2

    allowed = {item.strip() for item in args.classes.split(',') if item.strip()}
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(json.dumps({'status': 'blocked', 'reason': f'could not open video: {args.video}'}, ensure_ascii=False, indent=2))
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    rows = []
    frame_index = 0
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, args.frame_stride) != 0:
            frame_index += 1
            continue
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        names = result.names or getattr(model, 'names', {})
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
            confs = boxes.conf.detach().cpu().numpy().tolist()
            clss = boxes.cls.detach().cpu().numpy().astype(int).tolist()
            for bbox, conf, cls_id in zip(xyxy, confs, clss):
                class_name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
                if allowed and class_name not in allowed:
                    continue
                detections.append({'class_name': class_name, 'confidence': float(conf), 'bbox_xyxy': [float(v) for v in bbox]})
        rows.append({'video_id': Path(args.video).stem, 'frame_index': frame_index, 'timestamp_sec': frame_index / fps, 'detections': detections})
        used += 1
        frame_index += 1
    cap.release()
    write_jsonl(rows, args.output_jsonl)
    print(json.dumps({'video': args.video, 'model': args.model, 'output_jsonl': args.output_jsonl, 'sampled_frames': used, 'frame_stride': args.frame_stride}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
