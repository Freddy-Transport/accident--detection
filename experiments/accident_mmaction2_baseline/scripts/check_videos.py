#!/root/autodl-tmp/traffic_accident_rnd/.venv/bin/python
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from common import load_data_config, read_manifest, sample_clip_frames, save_frame_grid

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--per-class',type=int,default=20); ap.add_argument('--num-frames',type=int,default=8); ap.add_argument('--output-dir'); args=ap.parse_args()
    cfg=load_data_config(args.config); rows=read_manifest(cfg['manifest_path']); out=Path(args.output_dir or Path(cfg['experiment_root'])/'outputs'/'video_checks'); out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(int(cfg['random_seed'])); selected=[]
    for label in ['accident','non_accident']:
        cand=[r for r in rows if r['mapped_label']==label]; rng.shuffle(cand); selected += cand[:args.per_class]
    results=[]
    for r in selected:
        item={'sample_id':r['sample_id'],'label':r['mapped_label'],'ok':False,'video_path':r['video_path']}
        try:
            frames=sample_clip_frames(r['video_path'],int(r['start_frame']),int(r['end_frame']),args.num_frames,160); img=out/f"{r['sample_id']}.jpg"; save_frame_grid(frames,img,title=f"{r['mapped_label']} {r['sample_id']}"); item.update({'ok':True,'grid_path':str(img),'decoded_frames':len(frames)})
        except Exception as exc: item['error']=str(exc)
        results.append(item)
    report={'checked':len(results),'ok':sum(1 for r in results if r['ok']),'failed':[r for r in results if not r['ok']]}; (out/'check_videos_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report['ok']==report['checked'] else 1
if __name__=='__main__': raise SystemExit(main())
