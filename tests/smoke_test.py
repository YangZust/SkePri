"""CPU-only checks, requiring no benchmark data or network access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import numpy as np
import torch
import yaml
from main import load_config, make_model, make_loader, train, seed_all
from types import SimpleNamespace
from ensemble import fuse, STREAMS
from checkpoint import load_checkpoint
from unittest.mock import patch
import subprocess
import inspect
from feeders.feeder_ntu import Feeder
from feeders.feeder_ucla import Feeder as UCLAFeeder
import pickle
import ast
import json


def test_checkpoint_compatibility():
    calls=[]

    def modern(path, map_location=None, *, weights_only=False):
        calls.append((path, map_location, weights_only))
        return {'model': {}}

    def legacy(path, map_location=None, **pickle_load_args):
        assert not pickle_load_args, pickle_load_args
        calls.append((path, map_location))
        return {'model': {}}

    with patch('checkpoint.torch.load', modern):
        load_checkpoint('new.pth')
        load_checkpoint('new.pth', trusted=True)
    assert calls == [('new.pth', 'cpu', True)] * 2
    calls.clear()
    with patch('checkpoint.torch.load', legacy):
        try:
            load_checkpoint('unknown.pth')
        except ValueError:
            pass
        else:
            raise AssertionError('Untrusted legacy checkpoint was accepted')
        assert not calls
        load_checkpoint('own.pth', trusted=True)
    assert calls == [('own.pth', 'cpu')]
    calls.clear()

    def rejected(path, map_location=None, *, weights_only=False):
        calls.append(weights_only)
        raise pickle.UnpicklingError('Restricted loading rejected checkpoint')

    with patch('checkpoint.torch.load', rejected):
        try:
            load_checkpoint('rejected.pth', trusted=True)
        except pickle.UnpicklingError:
            pass
        else:
            raise AssertionError('Restricted-loader failure was suppressed')
    assert calls == [True], 'No unsafe fallback is allowed on modern PyTorch'
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'weights.pth'
        torch.save({'weight': torch.arange(3)}, path)
        assert torch.equal(load_checkpoint(path, trusted=True)['weight'], torch.arange(3))
        if 'weights_only' not in inspect.signature(torch.load).parameters:
            try:
                load_checkpoint(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Actual legacy loader did not enforce trust')
    print('Checkpoint compatibility: restricted loading / legacy trust / no unsafe fallback passed')


def test_raw_preprocessing():
    # Cover raw-file parsing, NumPy preprocessing, split construction, and the CLI.
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); raw=td/'raw'; raw.mkdir()
        names=('S002C002P001R001A001', 'S001C001P003R001A002',
               'S002C002P001R001A120', 'S001C002P005R002A008')
        lines=['3']
        for frame in range(3):
            lines.extend(['1', '1 0 0 0 0 0 0 0 0 0', '25'])
            lines.extend('{} {} {}'.format(0.1+joint*0.01, 1+joint*0.1, 2+frame*0.01)
                         for joint in range(25))
        for name in names:
            (raw/(name+'.skeleton')).write_text('\n'.join(lines)+'\n')
        for dataset, count, temporal_protocol in (('ntu60', 2, 'xview'), ('ntu120', 3, 'xset')):
            subprocess.run([sys.executable, 'data/prepare_ntu.py', '--dataset', dataset,
                            '--raw-dir', str(raw), '--output-dir', str(td/'prepared')], check=True)
            with (td/'prepared'/(dataset+'_3danno.pkl')).open('rb') as f:
                result=pickle.load(f)  # Generated locally by this test.
            assert len(result['annotations'])==count
            assert names[3] not in {a['frame_dir'] for a in result['annotations']}
            for protocol in ('xsub', temporal_protocol):
                assert names[0] in result['split'][protocol+'_train']
                assert result['split'][protocol+'_val']==[names[1]]
            for annotation in result['annotations']:
                assert annotation['keypoint'].shape==(1,3,25,3)
                assert np.isfinite(annotation['keypoint']).all()
    print('Synthetic raw NTU60/NTU120 preprocessing, exclusion and official splits passed')


def test_models():
    paths=sorted(Path('config').glob('*/*.yaml'))
    assert len(paths)==20
    torch.set_num_threads(2)
    checked=set()
    for p in paths:
        cfg=load_config(p); assert cfg['data']['train_repeat']==5, p; assert cfg['model']['aux_weight']==0.01, p; m=make_model(cfg); d=make_model(cfg,deploy=True)
        assert sum(x.numel() for x in m.parameters())-sum(x.numel() for x in d.parameters())==257
        assert not m.prior.requires_grad and m.prior.shape==(m.num_classes,m.num_point,m.num_point)
        assert torch.allclose(m.prior.sum(-1),torch.ones_like(m.prior.sum(-1)),atol=1e-6)
        # Confirm direct Eq. (6)-(8) numerical equivalence independently.
        f=torch.randn(2,m.num_person,256,3,m.num_point); g=f.mean(3).mean(1)
        z=torch.stack([g @ a for a in m.prior],dim=1)
        expected=m.auxiliary((z*g[:,None]).mean(-1)).squeeze(-1)
        assert torch.allclose(m.auxiliary_scores(f),expected,atol=1e-5)
        if cfg['dataset'] not in checked:
            x=torch.randn(2,m.num_person,8,m.num_point,3); y=torch.tensor([0,1]); m.train()
            loss,logits=m.loss(x,y); loss.backward()
            assert logits.shape==(2,m.num_classes) and torch.isfinite(loss)
            assert m.auxiliary.weight.grad is not None and m.blocks[0].gcn1.convs[0].conv3.weight.grad is not None
            assert m.prior.grad is None
            m.eval(); d.eval(); d.load_state_dict(m.inference_state_dict(),strict=True)
            with torch.no_grad(): assert torch.equal(m(x),d(x))
            # Make sure inference never invokes auxiliary computation.
            m.auxiliary_scores=lambda *_: (_ for _ in ()).throw(AssertionError('aux executed in eval'))
            with torch.no_grad(): m(x)
            checked.add(cfg['dataset'])
            print(cfg['dataset'],'forward/backward/deploy passed')
    print('All 20 YAML model configurations and prior checks passed')


def test_data_training():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); p=td/'data.pkl'; rng=np.random.default_rng(42)
        ann=[dict(frame_dir=f'sample{i}',label=i%2,total_frames=12,keypoint=rng.normal(size=(1 if i%2 else 2,12,25,3)).astype(np.float32)) for i in range(4)]
        with p.open('wb') as f: pickle.dump(dict(annotations=ann,split={'xsub_train':['sample0','sample1'],'xsub_val':['sample2','sample3']}),f)
        for stream in STREAMS:
            ds=Feeder(p,'xsub',False,stream,8,10)
            x,y,n=ds[0]; assert x.shape==(10,2,8,25,3) and np.isfinite(x).all()
        cfg=load_config('config/ntu60-xsub/joint.yaml'); cfg['data'].update(path=str(p),clip_len=8,workers=0); cfg['training'].update(epochs=1,batch_size=2); cfg['work_dir']=str(td/'run')
        assert len(make_loader(cfg,'train').dataset)==10
        assert len(make_loader(cfg,'val').dataset)==2
        assert len(make_loader(cfg,'test').dataset)==2
        seed_all(1); train(cfg,SimpleNamespace(baseline=False,init_weights=None),torch.device('cpu'))
        run=Path(cfg['work_dir']); assert all((run/n).exists() for n in ('best.pth','best_inference.pth','best_scores.npz','best_result.json'))
        state=load_checkpoint(run/'best_inference.pth', trusted=True); assert 'prior' not in state and not any(k.startswith('auxiliary.') for k in state)
        # Exercise the public test CLI, including its independent weight-loading path.
        subprocess.run([sys.executable, 'main.py', '--config', str(run/'config.yaml'),
                        '--phase', 'test', '--weights', str(run/'best_inference.pth'),
                        '--trust-checkpoint', '--device', 'cpu', '--workers', '0',
                        '--output', str(run/'cli_scores.npz')], check=True)
        with np.load(run/'best_scores.npz') as expected, np.load(run/'cli_scores.npz') as actual:
            for name in ('scores', 'labels', 'ids'):
                assert np.array_equal(expected[name], actual[name]), name
        print('Synthetic NTU 1-epoch train/validate/best/10-clip test passed')
        # Check actual UCLA split metadata and feeder path using tiny artificial JSON files.
        tree=ast.parse(Path('feeders/feeder_ucla.py').read_text())
        records=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='data_dict' for t in node.targets):
                records.extend(ast.literal_eval(node.value))
        u=td/'ucla'; u.mkdir(); payload=json.dumps({'skeletons':rng.normal(size=(2,20,3)).tolist()})
        for row in records: (u/(row['file_name']+'.json')).write_text(payload)
        for stream in ('joint','bone','motion','bone motion'):
            for split in ('train','val'):
                ds=UCLAFeeder(stream,split,root=u,window_size=52); x,y,n=ds[0]; assert x.shape==(1,1,52,20,3) and np.isfinite(x).all()
        cfg_u=load_config('config/nwucla/joint.yaml'); cfg_u['data'].update(path=str(u),workers=0)
        train_u=make_loader(cfg_u,'train').dataset; val_u=make_loader(cfg_u,'val').dataset; test_u=make_loader(cfg_u,'test').dataset
        assert len(train_u)==5*len(train_u.data_dict)
        assert len(val_u)==len(val_u.data_dict) and len(test_u)==len(test_u.data_dict)
        print('Synthetic UCLA feeder: all 4 streams, both splits, and train/eval repetition passed')
        # ID alignment: deliberately reverse alternate streams.
        paths=[]
        for i,stream in enumerate(STREAMS):
            idx=np.array([1,0]) if i%2 else np.array([0,1]); path=td/(stream+'.npz')
            np.savez(path,ids=np.array(['a','b'])[idx],labels=np.array([0,1])[idx],scores=np.eye(2)[idx],dataset='ntu60',protocol='xsub',stream=stream); paths.append(path)
        scores,labels,ids=fuse(paths,[1,2,3,4]); assert np.array_equal(scores.argmax(1),labels)
        subprocess.run([sys.executable, 'ensemble.py', '--joint', str(paths[0]),
                        '--bone', str(paths[1]), '--joint-motion', str(paths[2]),
                        '--bone-motion', str(paths[3]), '--weights', '1', '2', '3', '4',
                        '--output', str(td/'fused.npz')], check=True)
        with np.load(td/'fused.npz') as actual:
            assert np.array_equal(actual['scores'], scores)
        print('Four-stream ID-aligned fusion passed')

if __name__=='__main__':
    test_checkpoint_compatibility(); test_raw_preprocessing(); test_models(); test_data_training(); print('ALL SMOKE TESTS PASSED')
