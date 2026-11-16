"""Regression cases for release contracts; uses synthetic, local-only fixtures."""
import ast
import copy
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ensemble import fuse,STREAMS
from main import load_config,lr_at,train
from checkpoint import load_checkpoint
from provenance import resolve_metadata,sha256
from validation import validate_config,annotation_metadata
from data.prepare_ntu import gen_keypoint_array,spread_denoising,non_zero,MISSING
from feeders.feeder_ntu import Feeder
from feeders.feeder_ucla import Feeder as UCLAFeeder

def rejects(fn):
    try: fn()
    except (ValueError,TypeError): return
    raise AssertionError('Invalid input was accepted')

def test_contracts():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); paths=[td/(s+'.npz') for s in STREAMS]
        base=dict(ids=np.array(['a','b']),labels=np.array([0,1]),scores=np.eye(60)[:2],dataset='ntu60',protocol='xsub',method='skepri',seed=1,checkpoint_epoch=3,score_type='probability',schema_version=1,checkpoint_sha256='0'*64)
        def write(changes=None):
            for i,(p,s) in enumerate(zip(paths,STREAMS)):
                d=dict(base,stream=s,seed=i,checkpoint_epoch=i+2)
                if i==1 and changes: d.update(changes)
                np.savez(p,**d)
        write(); scores,y,ids=fuse(paths,[1,2,3,4]); assert np.array_equal(scores.argmax(1),y)
        assert np.array_equal(scores,np.tensordot(np.array([1,2,3,4])/10,np.stack([base['scores']]*4),axes=1))
        for patch in [dict(labels=np.array([[0],[1]])),dict(labels=np.array([0.,1.])),dict(labels=np.array([-1,1])),dict(labels=np.array([60,1])),dict(ids=np.array(['a','a'])),dict(ids=np.array(['','b'])),dict(ids=np.empty(0,dtype=str)),dict(ids=np.array([['a'],['b']])),dict(ids=np.array(['a','c'])),dict(scores=np.zeros((2,59))),dict(scores=np.full((2,60),np.nan)),dict(scores=np.full((2,60),np.inf)),dict(scores=np.full((2,60),'x')),dict(scores=np.ones((2,60),dtype=complex)),dict(dataset='ntu120'),dict(protocol='xview'),dict(method='ctrgcn_baseline'),dict(score_type='logit'),dict(stream='joint')]:
            write(patch); rejects(lambda:fuse(paths,[1,1,1,1]))
        write()
        for w in ([0,0,0,0],[-1,1,1,1],[np.nan,1,1,1],[np.inf,1,1,1],[1e308]*4,[1,1,1]): rejects(lambda:fuse(paths,w))
        rejects(lambda:fuse(paths[:3],[1]*4)); rejects(lambda:fuse(paths+[paths[0]],[1]*4))
        # Correctly aligned reversed samples, and all-baseline fusion.
        write(dict(ids=base['ids'][::-1],labels=base['labels'][::-1],scores=base['scores'][::-1]))
        assert np.array_equal(fuse(paths,[1]*4)[0],base['scores'])
        for p,s in zip(paths,STREAMS): np.savez(p,**dict(base,ids=np.array([10,20]),stream=s))
        assert np.array_equal(fuse(paths,[1]*4)[2],np.array(['10','20']))
        for p,s in zip(paths,STREAMS): np.savez(p,**dict(base,stream=s,method='ctrgcn_baseline'))
        assert np.array_equal(fuse(paths,[1]*4)[0],base['scores'])
        np.savez(paths[0],ids=base['ids'],labels=base['labels'],scores=base['scores'])
        rejects(lambda:fuse(paths,[1]*4))
        cfg=load_config('config/ntu60-xsub/joint.yaml')
        for group,key,value in [('evaluation','interval',5),('training','schedule','unknown'),('data','train_repeat',0),('training','lr',float('nan')),('evaluation','batch_size',0)]:
            c=copy.deepcopy(cfg); c[group][key]=value; rejects(lambda:validate_config(c))
        for p in Path('config').glob('*/*.yaml'):
            c=load_config(p); assert c['data']['train_repeat']==5 and c['model']['aux_weight']==.01
            for epoch in (0,4,5,49,50,64): assert np.isfinite(lr_at(c,epoch))
        ann=[dict(frame_dir='a',label=0,total_frames=12,keypoint=np.ones((1,12,25,3),dtype=np.float32)),dict(frame_dir='b',label=1,total_frames=12,keypoint=np.ones((1,12,25,3),dtype=np.float32))]
        data=dict(annotations=ann,split=dict(xsub_train=['a'],xsub_val=['b']))
        annotation_metadata(data,'ntu60','xsub',True)
        rejects(lambda:annotation_metadata(data,'ntu60','xsub'))
        for mutate in (lambda d:d['annotations'].append(d['annotations'][0]),lambda d:d['split']['xsub_train'].append('a'),lambda d:d['split']['xsub_val'].append('a'),lambda d:d['split']['xsub_val'].append('missing'),lambda d:d['annotations'][0].update(label=60),lambda d:d['annotations'][0].update(total_frames=13)):
            bad=copy.deepcopy(data); mutate(bad); rejects(lambda:annotation_metadata(bad,'ntu60','xsub',True))
        p=td/'data.pkl'
        for value in (np.zeros((1,12,25,3)),np.full((1,12,25,3),np.nan),np.full((1,12,25,3),np.inf)):
            bad=copy.deepcopy(data); bad['annotations'][0]['keypoint']=value
            with p.open('wb') as f: pickle.dump(bad,f)
            rejects(lambda:Feeder(p,'xsub',True,'joint',allow_partial=True)[0])
        rejects(lambda:gen_keypoint_array({}))
        bodies={i:dict(kpt=np.ones((3,25,3)),start=0,motion=1) for i in (1,2)}
        rejects(lambda:gen_keypoint_array(bodies)); rejects(lambda:non_zero(np.zeros((2,3,25,3))))
        wide=np.ones((12,25,3)); wide[:,:,0]=np.arange(25)
        rejects(lambda:spread_denoising([dict(kpt=wide,motion=1)]))
        rejects(lambda:UCLAFeeder('joint','train',normalization=True))
        rejects(lambda:UCLAFeeder('joint','train',root=td/'missing'))
        tree=ast.parse(Path('feeders/feeder_ucla.py').read_text(encoding='utf-8'))
        splits=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='data_dict' for t in n.targets)]
        assert sorted(map(len,splits))==[464,1020]
        sample=splits[0][0]['file_name']; (td/(sample+'.json')).write_text(json.dumps(dict(skeletons=[[1,2,3]])))
        # The first file is malformed, so no remaining split files are needed.
        split='val' if len(splits[0])==464 else 'train'
        rejects(lambda:UCLAFeeder('joint',split,root=td))
        for payload in ('not json',json.dumps({}),json.dumps(dict(skeletons=[])),json.dumps(dict(skeletons=np.full((2,20,3),np.nan).tolist())),json.dumps(dict(skeletons=np.full((2,20,3),np.inf).tolist()))):
            (td/(sample+'.json')).write_text(payload)
            rejects(lambda:UCLAFeeder('joint',split,root=td))
        # Baseline end-to-end and deployment provenance/hashes.
        with p.open('wb') as f: pickle.dump(data,f)
        cfg['data'].update(path=str(p),workers=0,clip_len=8,allow_partial=True)
        cfg['training'].update(epochs=1,batch_size=2); cfg['work_dir']=str(td/'baseline')
        torch.set_num_threads(2)
        train(cfg,SimpleNamespace(baseline=True,init_weights=None),torch.device('cpu'))
        run=Path(cfg['work_dir']); meta=json.loads((run/'best_inference.meta.json').read_text())
        assert meta['method']=='ctrgcn_baseline' and meta['checkpoint_epoch']==1
        assert meta['checkpoint_sha256']==sha256(run/'best_inference.pth')
        state=load_checkpoint(run/'best_inference.pth',trusted=True)
        args=SimpleNamespace(baseline=False,source_method=None)
        assert resolve_metadata(state,run/'best_inference.pth',cfg,args)['seed']==1
        assert resolve_metadata(load_checkpoint(run/'best.pth',trusted=True),run/'best.pth',cfg,args)['method']=='ctrgcn_baseline'
        with np.load(run/'best_scores.npz',allow_pickle=False) as f: assert f['method'].item()=='ctrgcn_baseline'
        subprocess.run([sys.executable,'main.py','--config',str(run/'config.yaml'),'--phase','test','--weights',str(run/'best_inference.pth'),'--trust-checkpoint','--device','cpu','--output',str(run/'again.npz')],check=True)
        with np.load(run/'again.npz') as a,np.load(run/'best_scores.npz') as b:
            assert np.array_equal(a['scores'],b['scores']) and a['method']==b['method']
        badcfg=copy.deepcopy(cfg); badcfg['stream']='bone'; rejects(lambda:resolve_metadata(state,run/'best_inference.pth',badcfg,args))
        meta['checkpoint_sha256']='f'*64; (run/'best_inference.meta.json').write_text(json.dumps(meta))
        rejects(lambda:resolve_metadata(state,run/'best_inference.pth',cfg,args))
        legacy=td/'legacy.pth'; torch.save(state,legacy)
        rejects(lambda:resolve_metadata(state,legacy,cfg,args))
        declared=SimpleNamespace(baseline=False,source_method='ctrgcn_baseline',source_seed=-1,source_epoch=-1)
        assert resolve_metadata(state,legacy,cfg,declared)['seed']==-1
    print('Release contracts: fusion, provenance, partial/invalid data, baseline lifecycle, config validation PASSED')
