"""Four-stream score fusion with explicit weights and sample-ID alignment."""
import argparse
from pathlib import Path
import numpy as np

STREAMS=('joint','bone','joint_motion','bone_motion')

def fuse(paths, weights):
    arrays=[]; ids=None; labels=None; protocol=None
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(4,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Four finite nonnegative weights, at least one positive, required')
    for path,stream in zip(paths,STREAMS):
        with np.load(path,allow_pickle=False) as f:
            if str(f['stream'])!=stream: raise ValueError(f'Expected {stream}: {path}')
            key=(str(f['dataset']),str(f['protocol']))
            names=f['ids'].astype(str); y=f['labels']; s=f['scores']
            if len(set(names))!=len(names) or s.ndim!=2 or len(y)!=len(names) or len(s)!=len(names) or not np.isfinite(s).all(): raise ValueError(f'Invalid scores: {path}')
            if ids is None: ids=names; labels=y; protocol=key
            if key!=protocol or set(names)!=set(ids): raise ValueError('Different datasets, protocols or sample IDs')
            mapping={n:i for i,n in enumerate(names)}; ix=[mapping[n] for n in ids]
            if not np.array_equal(y[ix],labels): raise ValueError('Label mismatch')
            arrays.append(s[ix])
    scores=np.tensordot(weights/weights.sum(),np.stack(arrays),axes=1)
    return scores, labels, ids

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for stream in STREAMS: p.add_argument('--'+stream.replace('_','-'),required=True)
    p.add_argument('--weights',nargs=4,type=float,required=True,metavar=('J','B','JM','BM'))
    p.add_argument('--output'); a=p.parse_args()
    s,y,ids=fuse([getattr(a,k) for k in STREAMS],a.weights)
    print(f'Four-stream Top-1: {100*(s.argmax(1)==y).mean():.4f}% (weights={a.weights})')
    if a.output:
        path=Path(a.output)
        if path.suffix!='.npz' or path.exists(): p.error('Choose a new .npz output path')
        path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,scores=s,labels=y,ids=ids)
if __name__=='__main__': main()
