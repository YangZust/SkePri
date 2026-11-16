"""Plain-data checkpoint provenance; deployment weights remain state_dicts."""
import hashlib
import json
from pathlib import Path
from validation import DATASETS,STREAMS,integer

METHODS=('skepri','ctrgcn_baseline')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def validate_metadata(m,cfg=None):
    required=('method','dataset','protocol','stream','seed','checkpoint_epoch','score_type','schema_version')
    if not isinstance(m,dict) or any(k not in m for k in required): raise ValueError('Missing/incomplete provenance metadata')
    if m['method'] not in METHODS or m['dataset'] not in DATASETS or m['stream'] not in STREAMS: raise ValueError('Invalid provenance identity')
    if m['protocol'] not in DATASETS[m['dataset']][2]: raise ValueError('Invalid provenance protocol')
    if m['score_type']!=('logit' if m['dataset']=='nwucla' else 'probability'): raise ValueError('Invalid score_type')
    integer(m['seed'],'training seed',-1); integer(m['checkpoint_epoch'],'checkpoint epoch',-1)
    integer(m['schema_version'],'schema_version')
    if m['schema_version']!=1: raise ValueError('Unsupported schema_version')
    if cfg and any(m[k]!=cfg[k] for k in ('dataset','protocol','stream')): raise ValueError('Checkpoint provenance conflicts with configuration')
    return dict(m)

def resolve_metadata(checkpoint,path,cfg,args):
    m=checkpoint.get('metadata') if isinstance(checkpoint,dict) else None
    sidecar=Path(path).with_suffix('.meta.json')
    if m is None and sidecar.exists(): m=json.loads(sidecar.read_text(encoding='utf-8'))
    declared=getattr(args,'source_method',None)
    if m is None:
        if not declared: raise ValueError('No checkpoint provenance: explicitly declare legacy --source-method')
        m=dict(method=declared,dataset=cfg['dataset'],protocol=cfg['protocol'],stream=cfg['stream'],seed=getattr(args,'source_seed',-1),checkpoint_epoch=getattr(args,'source_epoch',-1),score_type='logit' if cfg['dataset']=='nwucla' else 'probability',schema_version=1,provenance_source='user_declared_legacy')
    m=validate_metadata(m,cfg)
    if declared and declared!=m['method']: raise ValueError('Source declaration conflicts with checkpoint')
    if getattr(args,'baseline',False) and m['method']!='ctrgcn_baseline': raise ValueError('--baseline conflicts with checkpoint')
    digest=sha256(path)
    if 'checkpoint_sha256' in m and m['checkpoint_sha256']!=digest: raise ValueError('Checkpoint SHA-256 mismatch')
    m['checkpoint_sha256']=digest
    return m
