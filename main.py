"""Unified training, validation and deployment entry point for SkePri."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.skepri import Model
from feeders.feeder_ntu import Feeder as NTUFeeder
from feeders.feeder_ucla import Feeder as UCLAFeeder
from checkpoint import load_checkpoint
from validation import validate_config
from provenance import sha256, resolve_metadata, validate_metadata, METHODS


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return validate_config(cfg)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(cfg, phase):
    d, e = cfg['data'], cfg['evaluation']
    train = phase == 'train'
    clips = 1 if train else e['val_clips' if phase=='val' else 'test_clips']
    repeat = d['train_repeat'] if train else 1
    if cfg['dataset'] == 'nwucla':
        stream = {'joint':'joint','bone':'bone','joint_motion':'motion','bone_motion':'bone motion'}[cfg['stream']]
        ds = UCLAFeeder(data_path=stream,label_path='train' if train else 'val',repeat=repeat,window_size=d['clip_len'],root=d['path'])
    else:
        ds = NTUFeeder(d['path'],cfg['protocol'],train,cfg['stream'],d['clip_len'],clips,repeat,
                       dataset=cfg['dataset'],allow_partial=d.get('allow_partial',False),reference_path=d.get('reference_path'))
    return DataLoader(ds,batch_size=cfg['training']['batch_size'] if train else e['batch_size'],shuffle=train,drop_last=train,num_workers=d['workers'],pin_memory=torch.cuda.is_available())


def make_model(cfg, baseline=False, deploy=False):
    args = dict(cfg['model'])
    if baseline or deploy: args['prior_path'] = None
    return Model(dataset=cfg['dataset'], **args)


def lr_at(cfg, epoch):
    import math
    t=cfg['training']; lr=t['lr']; warm=t['warmup_epochs']
    if epoch < warm: return lr*(epoch+1)/warm
    if t['schedule']=='cosine': return lr*0.5*(1+math.cos(math.pi*epoch/t['epochs']))
    return lr*t['lr_decay']**sum(epoch>=step for step in t['milestones'])


@torch.inference_mode()
def evaluate(model, loader, cfg, device, desc):
    model.eval(); scores=[]; labels=[]; ids=[]
    for x,y,names in tqdm(loader,desc=desc,leave=False):
        x=x.to(device, non_blocking=True); n,c=x.shape[:2]
        # NTU PYSKL uses probability averaging over test clips. UCLA uses
        # original single-clip logits. No auxiliary branch is evaluated.
        # Evaluate clips one at a time to bound memory.
        logits=torch.stack([model(x[:,i]) for i in range(c)],dim=1)
        out=logits.mean(1) if cfg['dataset']=='nwucla' else logits.softmax(-1).mean(1)
        scores.append(out.cpu().numpy()); labels.append(y.numpy()); ids.extend(names)
    scores=np.concatenate(scores); labels=np.concatenate(labels)
    if len(set(ids))!=len(ids): raise ValueError('Repeated IDs in evaluation split')
    return float(100*(scores.argmax(1)==labels).mean()),dict(scores=scores,labels=labels,ids=np.asarray(ids,dtype=str))


def save_scores(path, payload, cfg, metadata):
    metadata=validate_metadata(metadata,cfg)
    with Path(path).open('xb') as f:
        np.savez_compressed(f,**payload,**metadata)


def train(cfg,args,device):
    validate_config(cfg)
    work=Path(cfg['work_dir']); work.mkdir(parents=True,exist_ok=False)
    (work/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    model=make_model(cfg,args.baseline).to(device)
    method='skepri' if model.auxiliary is not None else 'ctrgcn_baseline'
    metadata=dict(method=method,dataset=cfg['dataset'],protocol=cfg['protocol'],stream=cfg['stream'],
                  seed=cfg['seed'],checkpoint_epoch=-1,score_type='logit' if cfg['dataset']=='nwucla' else 'probability',schema_version=1)
    if args.init_weights:
        ckpt=load_checkpoint(args.init_weights, trusted=getattr(args, 'trust_checkpoint', False))
        state=ckpt.get('model',ckpt)
        # Initialization transfers only the main pathway. Never replace the
        # configured fixed prior or auxiliary initialization with legacy values.
        state={k:v for k,v in state.items() if k!='prior' and not k.startswith('auxiliary.')}
        incompatible=model.load_state_dict(state,strict=False)
        if incompatible.unexpected_keys or any(not (k.startswith('auxiliary.') or k=='prior') for k in incompatible.missing_keys):
            raise ValueError(f'Incompatible initialization: {incompatible}')
    t=cfg['training']
    opt=torch.optim.SGD(model.parameters(),lr=t['lr'],momentum=t['momentum'],nesterov=t['nesterov'],weight_decay=t['weight_decay'])
    train_loader=make_loader(cfg,'train'); val_loader=make_loader(cfg,'val')
    if not len(train_loader): raise ValueError('Training split smaller than batch size')
    best=-1.; history=[]
    for epoch in range(t['epochs']):
        model.train(); lr=lr_at(cfg,epoch)
        for group in opt.param_groups: group['lr']=lr
        total=correct=0; loss_sum=0.
        bar=tqdm(train_loader,desc=f'Epoch {epoch+1}/{t["epochs"]}')
        for x,y,_ in bar:
            x=x[:,0].to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); loss,logits=model.loss(x,y)
            loss.backward(); opt.step()
            total+=y.numel(); correct+=(logits.argmax(1)==y).sum().item(); loss_sum+=loss.item()*y.numel()
            bar.set_postfix(loss=f'{loss_sum/total:.4f}',acc=f'{100*correct/total:.2f}',lr=f'{lr:.5g}')
        acc,_=evaluate(model,val_loader,cfg,device,'Official test split (single clip)')
        history.append(dict(epoch=epoch+1,train_loss=loss_sum/total,train_top1=100*correct/total,val_top1=acc,lr=lr))
        (work/'history.json').write_text(json.dumps(history,indent=2))
        if acc>best:
            metadata['checkpoint_epoch']=epoch+1
            best=acc; torch.save(dict(model=model.state_dict(),epoch=epoch+1,val_top1=acc,metadata=dict(metadata)),work/'best.pth')
        print(f'Epoch {epoch+1}: Top-1 {acc:.4f}%; best {best:.4f}%',flush=True)
    # This checkpoint was created by this training run, so legacy loading is trusted.
    ckpt=load_checkpoint(work/'best.pth', trusted=True)
    model.load_state_dict(ckpt['model']); model.eval()
    torch.save(model.inference_state_dict(),work/'best_inference.pth')
    metadata=dict(ckpt['metadata'],checkpoint_sha256=sha256(work/'best_inference.pth'))
    (work/'best_inference.meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    # Deploy-only model: no prior or auxiliary weights in memory or checkpoint.
    deploy=make_model(cfg,deploy=True).to(device)
    deploy.load_state_dict(model.inference_state_dict(),strict=True)
    acc,payload=evaluate(deploy,make_loader(cfg,'test'),cfg,device,'Best checkpoint: final test')
    save_scores(work/'best_scores.npz',payload,cfg,metadata)
    result=dict(best_epoch=ckpt['epoch'],selection_top1=ckpt['val_top1'],final_top1=acc,metadata=metadata)
    (work/'best_result.json').write_text(json.dumps(result,indent=2))
    print(f'BEST checkpoint epoch {ckpt["epoch"]}: final Top-1 {acc:.4f}%')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--phase',choices=['train','test'],default='train')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); p.add_argument('--seed',type=int)
    p.add_argument('--work-dir'); p.add_argument('--data-path'); p.add_argument('--workers',type=int)
    p.add_argument('--baseline',action='store_true'); p.add_argument('--init-weights'); p.add_argument('--weights')
    p.add_argument('--allow-partial',action='store_true',help='Debug partial datasets only; never formal experiments')
    p.add_argument('--reference-data',help='Trusted official PYSKL PKL used to verify full sample IDs and splits')
    p.add_argument('--source-method',choices=METHODS,help='Explicit declaration for legacy checkpoints without provenance')
    p.add_argument('--source-seed',type=int,default=-1); p.add_argument('--source-epoch',type=int,default=-1)
    p.add_argument('--trust-checkpoint', action='store_true', help='Allow trusted --weights/--init-weights on old PyTorch without weights_only; newer PyTorch always uses weights_only=True')
    p.add_argument('--output',help='New .npz path for test scores'); p.add_argument('--validate',action='store_true',help='Validation is always performed every epoch')
    p.add_argument('--test-best',action='store_true',help='Best checkpoint is always tested at end of training')
    args=p.parse_args(); cfg=load_config(args.config)
    if args.seed is not None: cfg['seed']=args.seed
    if args.work_dir: cfg['work_dir']=args.work_dir
    if args.data_path: cfg['data']['path']=args.data_path
    if args.workers is not None: cfg['data']['workers']=args.workers
    if args.baseline: cfg['model']['prior_path']=None
    if args.allow_partial:
        cfg['data']['allow_partial']=True
        print('WARNING: partial-data DEBUG mode; not a certified full benchmark dataset')
    if args.reference_data: cfg['data']['reference_path']=args.reference_data
    validate_config(cfg)
    seed_all(cfg['seed']); device=torch.device(args.device)
    if args.phase=='train':
        if Path(cfg['work_dir']).exists(): p.error('work_dir already exists; choose a new --work-dir (no overwrite).')
        train(cfg,args,device)
    else:
        if not args.weights: p.error('--phase test requires explicit --weights')
        weights=Path(args.weights)
        if not weights.exists(): p.error(f'Checkpoint not found: {weights}')
        model=make_model(cfg,deploy=True).to(device)
        checkpoint=load_checkpoint(weights, trusted=args.trust_checkpoint)
        metadata=resolve_metadata(checkpoint,weights,cfg,args)
        state=checkpoint.get('model',checkpoint)
        state={k:v for k,v in state.items() if k!='prior' and not k.startswith('auxiliary.')}
        model.load_state_dict(state,strict=True)
        output=Path(args.output or Path(cfg['work_dir'])/'test_scores.npz')
        if output.suffix!='.npz' or output.exists(): p.error('Use a new .npz output path')
        output.parent.mkdir(parents=True,exist_ok=True)
        acc,payload=evaluate(model,make_loader(cfg,'test'),cfg,device,'Test checkpoint')
        save_scores(output,payload,cfg,metadata); print(f'Top-1 {acc:.4f}%')

if __name__=='__main__': main()
