#!/root/autodl-tmp/traffic_accident_rnd/.venv/bin/python
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import torch
from torch import nn
from common import ensure_dir, load_data_config, load_experiment_config, read_manifest, sample_clip_frames
class TinyVideoClassifier(nn.Module):
    def __init__(self,n=2): super().__init__(); self.net=nn.Sequential(nn.Conv3d(3,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool3d((1,1,1)),nn.Flatten(),nn.Linear(16,n))
    def forward(self,x): return self.net(x)
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='videomae'); ap.add_argument('--max-batches',type=int,default=10); ap.add_argument('--backend',choices=['native_tiny','mmaction_probe'],default='native_tiny'); args=ap.parse_args()
    dc=load_data_config(); ec=load_experiment_config(); mc=ec['models'][args.model]; rows=read_manifest(dc['manifest_path']); rows=[r for r in rows if r.get('split')=='train'][:max(args.max_batches,1)]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); report={'model':args.model,'backend':args.backend,'device':str(device),'batches':[]}
    if args.backend=='mmaction_probe':
        try:
            import mmaction,mmcv,mmengine,decord
            report['mmaction_import']='ok'
        except Exception as exc:
            report['mmaction_import']=f'failed: {exc}'; print(json.dumps(report,ensure_ascii=False,indent=2)); return 2
    model=TinyVideoClassifier(int(ec['class_num'])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); loss_fn=nn.CrossEntropyLoss(); t0=time.perf_counter()
    for r in rows:
        frames=sample_clip_frames(r['video_path'],int(r['start_frame']),int(r['end_frame']),int(mc['clip_len']),int(mc['input_size']))
        x=torch.from_numpy(frames).float().permute(3,0,1,2).unsqueeze(0).to(device)/255.0; y=torch.tensor([int(r['label_id'])],device=device)
        opt.zero_grad(set_to_none=True); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
        report['batches'].append({'sample_id':r['sample_id'],'label':int(r['label_id']),'tensor_shape':list(x.shape),'logits_shape':list(logits.shape),'loss':float(loss.detach().cpu())})
    report['loss_is_finite']=all(math.isfinite(b['loss']) for b in report['batches']); report['elapsed_sec']=round(time.perf_counter()-t0,3)
    if device.type=='cuda': report['gpu_max_memory_mb']=round(torch.cuda.max_memory_allocated()/1024/1024,2)
    out=ensure_dir(Path(ec['outputs_root'])/f'{args.model}_sanity'); (out/'sanity_check_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'sanity_check_report.md').write_text(f"# Sanity Check Report\n\n- Model target: `{args.model}`\n- Backend: `{args.backend}`\n- Device: `{device}`\n- Batches: `{len(report['batches'])}`\n- Loss finite: `{report['loss_is_finite']}`\n- GPU max memory MB: `{report.get('gpu_max_memory_mb','n/a')}`\n\n",encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report['loss_is_finite'] and report['batches'] else 1
if __name__=='__main__': raise SystemExit(main())
