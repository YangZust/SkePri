"""Optional API generation; accepts original structured prompts or legacy lists."""
import argparse
import json
import math
import re
import sys
from pathlib import Path
import numpy as np

SIZES={'ntu60':(60,25),'ntu120':(120,25),'nwucla':(10,20)}
FORMAL_ROOT=Path(__file__).resolve().parents[1]/'priors'


def validate_matrix(value,nodes):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(nodes,nodes) or not np.isfinite(a).all() or a.min()<0 or a.max()>1 or not np.allclose(np.diag(a),1,rtol=0,atol=1e-7):
        raise ValueError('Expected finite {}x{} matrix in [0,1] with unit diagonal'.format(nodes,nodes))
    return a


def load_prompt_config(path,dataset,model=None,temperature=None):
    if dataset not in SIZES: raise ValueError('Unsupported dataset')
    k,v=SIZES[dataset]; raw=json.loads(Path(path).read_text(encoding='utf-8')); settings={}
    if isinstance(raw,list):
        prompts=raw; system=None
        print('Legacy user-only prompt list; not the original structured NTU60 request protocol')
    elif isinstance(raw,dict):
        if raw.get('dataset')!=dataset: raise ValueError('Prompt dataset mismatch')
        prompts=raw.get('prompts'); system=raw.get('system_prompt')
        if not isinstance(system,str) or not system.strip(): raise ValueError('Nonempty system_prompt required')
        for key,size in [('class_names',k),('joint_names',v)]:
            if key in raw and (not isinstance(raw[key],list) or len(raw[key])!=size or any(not isinstance(s,str) or not s.strip() for s in raw[key]) or len(set(raw[key]))!=size): raise ValueError('Invalid '+key)
        if all(key in raw for key in ('user_prompt_template','joints_info','class_names')):
            if not isinstance(raw['user_prompt_template'],str) or not isinstance(raw['joints_info'],str): raise ValueError('Invalid template/joints_info')
            expected=[raw['user_prompt_template'].format(JOINTS_INFO=raw['joints_info'],action=a) for a in raw['class_names']]
            if prompts!=expected: raise ValueError('Prompts differ from recorded template construction')
        settings=raw
    else: raise ValueError('Expected prompt list or structured object')
    if not isinstance(prompts,list) or len(prompts)!=k or any(not isinstance(s,str) or not s.strip() or re.search(r'\{[^{}]+\}',s) for s in prompts): raise ValueError('Invalid count, empty prompt or unresolved placeholders')
    for name in (settings.get('model','gpt-4o'),model if model is not None else 'gpt-4o'):
        if not isinstance(name,str) or not name.strip(): raise ValueError('Invalid model identifier')
    for temp in (settings.get('temperature',.1),temperature if temperature is not None else .1):
        if isinstance(temp,bool) or not isinstance(temp,(int,float)) or not math.isfinite(temp) or not 0<=temp<=2: raise ValueError('Invalid temperature')
    return dict(dataset=dataset,model=model if model is not None else settings.get('model','gpt-4o'),temperature=temperature if temperature is not None else settings.get('temperature',.1),system_prompt=system,prompts=prompts)


def build_messages(settings,prompt):
    messages=[]
    if settings['system_prompt'] is not None: messages.append(dict(role='system',content=settings['system_prompt']))
    messages.append(dict(role='user',content=prompt))
    return messages


def validate_output(output):
    out=Path(output)
    formal={FORMAL_ROOT/n for n in ('ntu60.npy','ntu120.npy','nwucla_joint.npy','nwucla_bone.npy')}
    if out.resolve() in {p.resolve() for p in formal}: raise ValueError('Formal prior paths are protected')
    if out.suffix!='.npy' or out.exists(): raise ValueError('Specify a new .npy output path')
    return out


def generate_bank(client,settings,output,retries=3):
    out=validate_output(output)
    if isinstance(retries,bool) or not isinstance(retries,int) or retries<1: raise ValueError('retries must be positive integer')
    _,v=SIZES[settings['dataset']]; results=[]
    print('Effective model={}, temperature={}'.format(settings['model'],settings['temperature']))
    for i,prompt in enumerate(settings['prompts']):
        for attempt in range(retries):
            try:
                response=client.chat.completions.create(model=settings['model'],temperature=settings['temperature'],messages=build_messages(settings,prompt))
                content=response.choices[0].message.content.strip()
                if content.startswith('```'): content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
                results.append(validate_matrix(json.loads(content),v)); break
            except (ValueError,TypeError,AttributeError,IndexError) as exc:
                if attempt==retries-1: raise RuntimeError('Class {}: no valid matrix'.format(i)) from exc
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f: np.save(f,np.stack(results),allow_pickle=False)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',required=True); p.add_argument('--dataset',choices=tuple(SIZES),required=True)
    p.add_argument('--output',required=True); p.add_argument('--model'); p.add_argument('--temperature',type=float)
    p.add_argument('--retries',type=int,default=3); a=p.parse_args()
    settings=load_prompt_config(a.prompts,a.dataset,a.model,a.temperature); validate_output(a.output)
    if a.retries<1: p.error('--retries must be positive')
    from openai import OpenAI
    print('Saved',generate_bank(OpenAI(),settings,a.output,a.retries))

if __name__=='__main__': main()
