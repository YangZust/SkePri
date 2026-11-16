"""Four-stream fusion with strict input contracts and source identity."""
import argparse
import json
from pathlib import Path
import numpy as np
from validation import DATASETS, STREAMS, real_array
from provenance import validate_metadata


def validate_score_file(path, stream):
    try:
        with np.load(path, allow_pickle=False) as f:
            required=('dataset','protocol','stream','ids','labels','scores','method','seed','checkpoint_epoch','score_type','schema_version','checkpoint_sha256')
            if any(k not in f for k in required): raise ValueError('Missing provenance; regenerate scores with main.py')
            meta={}
            for key in required:
                if key in ('ids','labels','scores'): continue
                if f[key].ndim!=0: raise ValueError('Metadata must be scalar: '+key)
                meta[key]=f[key].item()
            validate_metadata(meta)
            if meta['stream']!=stream: raise ValueError('Unexpected stream')
            names,y,s=f['ids'],f['labels'],f['scores']
            if names.ndim!=1 or not len(names) or names.dtype.kind not in 'USiufb': raise ValueError('ids must be nonempty 1D string-representable values')
            if names.dtype.kind in 'iufb' and not np.isfinite(names).all(): raise ValueError('Nonfinite IDs')
            names=names.astype(str)
            if any(not n.strip() for n in names) or len(set(names))!=len(names): raise ValueError('Empty/duplicate IDs')
            if y.ndim!=1 or y.dtype.kind not in 'iu' or len(y)!=len(names): raise ValueError('labels must be 1D integers matching IDs')
            s=real_array(s,'scores')
            if s.ndim!=2 or s.shape!=(len(names),DATASETS[meta['dataset']][0]): raise ValueError('Score dimensions/class count mismatch')
            if (y<0).any() or (y>=s.shape[1]).any(): raise ValueError('Labels out of range')
            digest=meta['checkpoint_sha256']
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('Invalid checkpoint hash')
            for key in ('provenance_source','prior_sha256'):
                if key in f:
                    if f[key].ndim!=0 or f[key].dtype.kind not in 'US': raise ValueError('Invalid optional provenance '+key)
                    meta[key]=f[key].item()
            return s,y,names,meta
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('{}: {}'.format(path,exc)) from exc


def fuse(paths, weights, return_metadata=False):
    paths=list(paths)
    if len(paths)!=4: raise ValueError('Exactly four score files required')
    weights=real_array(weights,'weights').astype(float)
    if weights.shape!=(4,) or (weights<0).any(): raise ValueError('Four nonnegative weights required')
    with np.errstate(over='ignore'): total=weights.sum()
    if not np.isfinite(total) or total<=0: raise ValueError('Weight sum must be finite and positive')
    arrays=[]; sources=[]; ids=None; labels=None; identity=None
    for path,stream in zip(paths,STREAMS):
        s,y,names,meta=validate_score_file(path,stream)
        key=tuple(meta[k] for k in ('dataset','protocol','method','score_type'))
        if ids is None: ids=names; labels=y; identity=key
        if key!=identity or set(names)!=set(ids): raise ValueError('Method/dataset/protocol/score type/sample set mismatch')
        mapping={n:i for i,n in enumerate(names)}; ix=[mapping[n] for n in ids]
        if not np.array_equal(y[ix],labels): raise ValueError('Label mismatch')
        arrays.append(s[ix]); sources.append(meta)
    scores=np.tensordot(weights/weights.sum(),np.stack(arrays),axes=1)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite fused scores')
    result=(scores,labels,ids)
    if return_metadata:
        result+=(dict(dataset=identity[0],protocol=identity[1],method=identity[2],score_type=identity[3],schema_version=1,fusion_weights=weights,stream_sources=json.dumps(sources,sort_keys=True)),)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for stream in STREAMS: p.add_argument('--'+stream.replace('_','-'),required=True)
    p.add_argument('--weights',nargs=4,type=float,required=True,metavar=('J','B','JM','BM'))
    p.add_argument('--output'); a=p.parse_args()
    s,y,ids,meta=fuse([getattr(a,k) for k in STREAMS],a.weights,True)
    print('Four-stream Top-1: {:.4f}% (method={}, weights={})'.format(100*(s.argmax(1)==y).mean(),meta['method'],a.weights))
    if a.output:
        path=Path(a.output)
        if path.suffix!='.npz' or path.exists(): p.error('Choose a new .npz output path')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: np.savez_compressed(f,scores=s,labels=y,ids=ids,**meta)

if __name__=='__main__': main()
