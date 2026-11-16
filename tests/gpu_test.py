"""Exact formal NTU120 input-size functional probe; no benchmark data."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import subprocess
import argparse
from main import load_config,make_model

def main():
    if not torch.cuda.is_available():
        print(json.dumps(dict(status='NOT_RUN',reason='CUDA GPU unavailable'))); return 2
    cfg=load_config('config/ntu120-xsub/joint.yaml')
    result=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                input_shape=[64,2,100,25,3],status='STARTED')
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(result,stage='before_forward')),flush=True)
    try:
        m=make_model(cfg).cuda().train(); x=torch.randn(64,2,100,25,3,device='cuda'); y=torch.arange(64,device='cuda')%120
        loss,_=m.loss(x,y); result['loss']=loss.item()
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        print(json.dumps(dict(result,stage='forward_passed',peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)
        loss.backward(); result['backward']='PASSED'; m.eval()
        with torch.no_grad():
            logits=m(x)
            if logits.shape!=(64,120) or not torch.isfinite(logits).all(): raise ValueError('Invalid inference output')
        torch.cuda.synchronize(); result['status']='PASSED'
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower(): raise
        result['status']='CUDA_OOM'; result['reason']='Exact formal dimensions exceed available GPU memory; no dimensions changed'
    finally:
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(); result['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASSED' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--timeout-seconds',type=int,default=180)
    args=parser.parse_args()
    if args.worker: sys.exit(main())
    if args.timeout_seconds<1: parser.error('timeout must be positive')
    try:
        completed=subprocess.run([sys.executable,'-u',__file__,'--worker'],timeout=args.timeout_seconds)
        sys.exit(completed.returncode)
    except subprocess.TimeoutExpired:
        print(json.dumps(dict(status='TIME_LIMIT',timeout_seconds=args.timeout_seconds,
                              reason='Exact formal dimensions did not finish; no pass/OOM claim; final peak unavailable')),flush=True)
        sys.exit(3)
