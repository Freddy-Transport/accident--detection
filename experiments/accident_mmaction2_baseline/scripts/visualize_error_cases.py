#!/root/autodl-tmp/traffic_accident_rnd/.venv/bin/python
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from common import ensure_dir, sample_clip_frames, save_frame_grid
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--max-per-kind',type=int,default=5); args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.predictions).open(newline='',encoding='utf-8'))); buckets={'tp_accident':[],'fp':[],'fn_accident':[],'tn':[],'low_confidence':[]}
    for r in rows:
        y=int(r['label_id']); p=int(r['pred_label']); s=float(r.get('prob_accident') or 0.0)
        if y==1 and p==1: buckets['tp_accident'].append(r)
        if y==0 and p==1: buckets['fp'].append(r)
        if y==1 and p==0: buckets['fn_accident'].append(r)
        if y==0 and p==0: buckets['tn'].append(r)
        if 0.4<=s<=0.6: buckets['low_confidence'].append(r)
    out=ensure_dir(Path(args.output_dir)/'error_cases'); lines=['# Error Analysis','']
    for kind,items in buckets.items():
        lines.append(f'## {kind}')
        for r in items[:args.max_per_kind]:
            try:
                frames=sample_clip_frames(r['video_path'],int(float(r['start_frame'])),int(float(r['end_frame'])),8,160); img=out/f"{kind}_{r['sample_id']}.jpg"; save_frame_grid(frames,img,title=f"{kind} true={r['label_id']} pred={r['pred_label']} score={r.get('prob_accident')}"); lines.append(f"- `{r['sample_id']}` grid=`{img}`")
            except Exception as exc: lines.append(f"- `{r.get('sample_id')}` failed: `{exc}`")
        lines.append('')
    Path(args.output_dir,'error_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps({k:len(v) for k,v in buckets.items()},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
