#!/root/autodl-tmp/traffic_accident_rnd/.venv/bin/python
from __future__ import annotations
import argparse, csv, json, subprocess
import cv2
from collections import Counter
from pathlib import Path
from common import ensure_dir, load_data_config, read_manifest, split_train_val

def clip_video_cv2(src: str, dst: Path, fps: float, start_frame: int, end_frame: int) -> bool:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_fps = fps if fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_idx = max(0, start_frame)
    ok_count = 0
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        ok_count += 1
        frame_idx += 1
    writer.release()
    cap.release()
    return ok_count > 0 and dst.exists() and dst.stat().st_size > 0

def clip_video(src: str, dst: Path, fps: float, start_frame: int, end_frame: int) -> bool:
    if dst.exists() and dst.stat().st_size > 0: return True
    ensure_dir(dst.parent); start=max(0.0,start_frame/max(fps,1e-6)); dur=max(0.1,(end_frame-start_frame)/max(fps,1e-6))
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',f'{start:.3f}','-i',src,'-t',f'{dur:.3f}','-c:v','libx264','-preset','veryfast','-crf','28','-an',str(dst)]
    try:
        if subprocess.run(cmd, check=False).returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return True
    except FileNotFoundError:
        pass
    return clip_video_cv2(src, dst, fps, start_frame, end_frame)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--max-per-split',type=int,default=40); ap.add_argument('--no-materialize',action='store_true'); args=ap.parse_args()
    cfg=load_data_config(args.config); rows=read_manifest(cfg['manifest_path']); train_ids,val_ids=split_train_val(rows,float(cfg['val_fraction_from_train']),int(cfg['random_seed']))
    buckets={'train':[],'val':[],'test':[]}
    for r in rows:
        dst='val' if r['sample_id'] in val_ids else 'train' if r['sample_id'] in train_ids else 'test' if r.get('split')=='test' else None
        if dst: buckets[dst].append(r)
    rng_seed=int(cfg['random_seed'])
    import random; rng=random.Random(rng_seed)
    ann_dir=Path(cfg['annotation_dir']); clip_root=Path(cfg['clip_root']); summary={}
    for split, split_rows in buckets.items():
        by_label={0:[],1:[]}
        for r in split_rows: by_label[int(r['label_id'])].append(r)
        selected=[]
        for label, items in by_label.items():
            rng.shuffle(items); selected.extend(items[:args.max_per_split] if args.max_per_split else items)
        ann_path=ann_dir/f'{split}.txt'; ensure_dir(ann_path.parent); written=[]; failed=[]
        with ann_path.open('w',encoding='utf-8') as f:
            for r in selected:
                clip_path=clip_root/split/f"{r['sample_id']}.mp4"
                ok=True if args.no_materialize else clip_video(r['video_path'],clip_path,float(r['fps']),int(r['start_frame']),int(r['end_frame']))
                if not ok: failed.append(r['sample_id']); continue
                f.write(f"{clip_path} {r['label_id']}\n"); written.append({**r,'clip_path':str(clip_path),'annotation_split':split})
        clip_csv=ann_dir/f'{split}_clips.csv'
        if written:
            fields=list(written[0].keys())
            with clip_csv.open('w',newline='',encoding='utf-8') as cf:
                w=csv.DictWriter(cf,fieldnames=fields); w.writeheader(); w.writerows(written)
        summary[split]={'selected':len(selected),'written':len(written),'failed':failed[:20],'labels':dict(Counter(r['mapped_label'] for r in written)),'ann_file':str(ann_path),'clip_csv':str(clip_csv)}
    (ann_dir/'annotation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if all(not v['failed'] for v in summary.values()) else 1
if __name__=='__main__': raise SystemExit(main())
