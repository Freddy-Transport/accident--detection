#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from common import ensure_dir, load_experiment_config, timestamp

def has_mmaction():
    try:
        import mmaction, mmengine, mmcv
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='videomae'); ap.add_argument('--work-dir'); ap.add_argument('--checkpoint'); args=ap.parse_args(); cfg=load_experiment_config(); ok,msg=has_mmaction()
    if not ok:
        print(json.dumps({'status':'blocked','reason':'MMAction2 import failed: '+msg},ensure_ascii=False,indent=2)); return 2
    mmaction_root=Path('/root/autodl-tmp/traffic_accident_rnd/third_party/mmaction2'); tool='train.py' if 'train_' in Path(__file__).name else 'test.py'; tool_path=mmaction_root/'tools'/tool
    if not tool_path.exists(): print(json.dumps({'status':'blocked','reason':f'Missing MMAction2 tool: {tool_path}'},ensure_ascii=False,indent=2)); return 2
    work=Path(args.work_dir or Path(cfg['outputs_root'])/f'{timestamp()}_{args.model}'); ensure_dir(work); cmd=[sys.executable,str(tool_path),cfg['models'][args.model]['config'],'--work-dir',str(work)]
    if tool=='test.py' and args.checkpoint: cmd.append(args.checkpoint)
    print(json.dumps({'status':'running','cmd':cmd},ensure_ascii=False,indent=2)); return subprocess.run(cmd).returncode
if __name__=='__main__': raise SystemExit(main())
