#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations
import csv, json, random, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f: return yaml.safe_load(f) or {}
def load_data_config(path: str | Path | None = None) -> dict[str, Any]: return load_yaml(path or EXPERIMENT_ROOT/'configs'/'accident_data.yaml')
def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]: return load_yaml(path or EXPERIMENT_ROOT/'configs'/'experiment.yaml')
def ensure_dir(path: str | Path) -> Path:
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p
def timestamp() -> str: return datetime.now().strftime('%Y%m%d_%H%M%S')
def read_metadata_rows(path: str | Path) -> list[dict[str,str]]:
    with Path(path).open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def read_manifest(path: str | Path) -> list[dict[str,str]]:
    with Path(path).open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def write_csv(path: str | Path, rows: list[dict[str,Any]], fields: list[str]) -> None:
    p=Path(path); ensure_dir(p.parent)
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); [w.writerow({k:r.get(k,'') for k in fields}) for r in rows]
def video_metadata(path: str | Path) -> dict[str, Any]:
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return {'opened':False,'fps':0.0,'num_frames':0,'width':0,'height':0,'duration':0.0}
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0); frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0); cap.release()
    return {'opened':True,'fps':fps,'num_frames':frames,'width':width,'height':height,'duration':frames/fps if fps>0 and frames else 0.0}
def sample_clip_frames(video_path: str | Path, start_frame: int, end_frame: int, num_frames: int, size: int=224) -> np.ndarray:
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f'could not open video: {video_path}')
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); start=max(0,min(start_frame,max(total-1,0))); end=max(start+1,min(end_frame,total if total else start+1))
    indices=np.linspace(start,max(start,end-1),num_frames).astype(int); frames=[]; last=None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
        if not ok:
            if last is None: continue
            frame=last.copy()
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA); last=rgb; frames.append(rgb)
    cap.release()
    if not frames: raise ValueError(f'no frames decoded: {video_path}')
    while len(frames)<num_frames: frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames],axis=0)
def save_frame_grid(frames: np.ndarray, output_path: str | Path, cols: int=4, title: str | None=None) -> None:
    h,w=frames[0].shape[:2]; rows=int(np.ceil(len(frames)/cols)); canvas=np.zeros((rows*h,cols*w,3),dtype=np.uint8)
    for i,frame in enumerate(frames):
        y=(i//cols)*h; x=(i%cols)*w; bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR); cv2.putText(bgr,str(i),(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2); canvas[y:y+h,x:x+w]=bgr
    if title: cv2.putText(canvas,title[:120],(8,canvas.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    p=Path(output_path); ensure_dir(p.parent); cv2.imwrite(str(p),canvas)
def split_train_val(rows: list[dict[str,Any]], val_fraction: float, seed: int) -> tuple[set[str],set[str]]:
    ids=[r['sample_id'] for r in rows if r.get('split')=='train']; rng=random.Random(seed); rng.shuffle(ids); n=max(1,int(round(len(ids)*val_fraction))) if ids else 0; return set(ids[n:]), set(ids[:n])
