"""Strict input contracts; no transformations of valid inputs."""
import numbers
import math
import numpy as np

DATASETS={'ntu60':(60,2,('xsub','xview')), 'ntu120':(120,2,('xsub','xset')), 'nwucla':(10,1,('xview',))}
STREAMS=('joint','bone','joint_motion','bone_motion')

def integer(x,name,minimum=1):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Integral) or x<minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))

def real_array(x,name):
    a=np.asarray(x)
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all(): raise ValueError(name+': expected finite real numbers')
    return a

def skeleton(x,joints,name):
    a=real_array(x,name)
    if a.ndim!=4 or not 1<=a.shape[0]<=2 or a.shape[1]<1 or a.shape[2:]!=(joints,3):
        raise ValueError(name+': expected [M,T,'+str(joints)+',3], 1<=M<=2, T>0')
    if not np.any(a!=0): raise ValueError(name+': all-zero skeleton / no valid frames')
    return a

def validate_config(c):
    try:
        k,m,protocols=DATASETS[c['dataset']]; d,t,e=c['data'],c['training'],c['evaluation']
        integer(c['model']['num_classes'],'num_classes'); integer(c['model']['num_person'],'num_person')
        if c['protocol'] not in protocols or c['stream'] not in STREAMS: raise ValueError('Invalid dataset/protocol/stream')
        if (c['model']['num_classes'],c['model']['num_person'])!=(k,m): raise ValueError('Class/person mismatch')
        for n,x in [('seed',c['seed']),('workers',d['workers']),('warmup_epochs',t['warmup_epochs'])]: integer(x,n,0)
        for n,x in [('train_repeat',d['train_repeat']),('clip_len',d['clip_len']),('epochs',t['epochs']),('batch_size',t['batch_size']),('eval batch_size',e['batch_size']),('val_clips',e['val_clips']),('test_clips',e['test_clips'])]: integer(x,n)
        integer(e['interval'],'evaluation.interval')
        if e['interval']!=1: raise ValueError('SkePri requires evaluation.interval=1.')
        if t['schedule'] not in ('cosine','step'): raise ValueError('Unsupported learning-rate schedule')
        for n,x,positive in [('lr',t['lr'],True),('momentum',t['momentum'],False),('weight_decay',t['weight_decay'],False),('lr_decay',t['lr_decay'],True),('aux_weight',c['model']['aux_weight'],False),('dropout',c['model']['dropout'],False)]:
            if isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) or x<0 or (positive and x==0): raise ValueError('Invalid '+n)
        if c['model']['dropout']>=1 or t['momentum']>=1 or t['warmup_epochs']>t['epochs']: raise ValueError('Invalid dropout/momentum/warm-up')
        if not isinstance(t['nesterov'],bool) or (t['nesterov'] and t['momentum']<=0): raise ValueError('Invalid Nesterov setting')
        for step in t['milestones']: integer(step,'milestone',0)
    except (KeyError,TypeError) as exc: raise ValueError('Invalid configuration: '+str(exc)) from exc
    return c

def annotation_metadata(data,dataset,protocol,allow_partial=False,reference=None):
    if dataset not in ('ntu60','ntu120') or protocol not in DATASETS[dataset][2]: raise ValueError('Invalid NTU dataset/protocol')
    if not isinstance(data,dict) or not isinstance(data.get('split'),dict) or not isinstance(data.get('annotations'),list): raise ValueError('Expected PYSKL split/annotations')
    by_id={}
    for a in data['annotations']:
        if not isinstance(a,dict) or not isinstance(a.get('frame_dir'),str) or not a['frame_dir'].strip(): raise ValueError('Invalid annotation ID')
        n=a['frame_dir']
        if n in by_id: raise ValueError('Duplicate annotation ID: '+n)
        integer(a.get('label'),n+' label',0); integer(a.get('total_frames'),n+' total_frames')
        if a['label']>=DATASETS[dataset][0]: raise ValueError(n+': label out of range')
        kp=a.get('keypoint')
        if not isinstance(kp,np.ndarray) or kp.ndim!=4 or not 1<=kp.shape[0]<=2 or kp.shape[1:]!=(a['total_frames'],25,3) or kp.dtype.kind not in 'iuf': raise ValueError(n+': invalid shape/dtype/total_frames')
        by_id[n]=a
    sets=[]
    for suffix in ('_train','_val'):
        ids=data['split'].get(protocol+suffix)
        if not isinstance(ids,(list,tuple)) or not ids or any(not isinstance(n,str) or not n.strip() for n in ids): raise ValueError('Missing/empty/invalid split '+protocol+suffix)
        if len(set(ids))!=len(ids): raise ValueError('Duplicate split IDs')
        if not set(ids)<=set(by_id): raise ValueError('Split IDs missing annotations')
        sets.append(set(ids))
    if sets[0]&sets[1]: raise ValueError('Training/test split overlap')
    if sets[0]|sets[1]!=set(by_id): raise ValueError('Protocol does not cover all annotations')
    if not allow_partial:
        if {a['label'] for a in by_id.values()}!=set(range(DATASETS[dataset][0])): raise ValueError('Incomplete class coverage; --allow-partial is debugging only')
        if reference is None: raise ValueError('Completeness NOT certified: supply a trusted official --reference-data/--reference PKL; expected counts are not assumed')
        refs=annotation_metadata(reference,dataset,protocol,True)
        if set(refs)!=set(by_id): raise ValueError('Sample IDs differ from trusted reference')
        if any(a['label']!=refs[n]['label'] for n,a in by_id.items()): raise ValueError('Reference label mismatch')
        for suffix in ('_train','_val'):
            if set(data['split'][protocol+suffix])!=set(reference['split'][protocol+suffix]): raise ValueError('Split differs from reference')
    return by_id
