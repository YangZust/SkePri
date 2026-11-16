"""Optional offline prior generation. Not imported by training or inference.

Input: a JSON array of complete, dataset-specific prompts in class-index order.
Use the actual prompt templates/settings and joint mapping from supplementary
material. No historical prompt is reconstructed or invented by this script.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def validate_matrix(value, nodes):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(nodes,nodes) or not np.isfinite(a).all() or a.min()<0 or a.max()>1 or not np.allclose(np.diag(a),1,rtol=0,atol=1e-7):
        raise ValueError(f'Expected a finite {nodes}x{nodes} matrix in [0,1] with unit diagonal')
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',required=True,help='JSON array of complete prompt strings, in class-index order')
    p.add_argument('--dataset',required=True,choices=['ntu60','ntu120','nwucla'])
    p.add_argument('--output',required=True); p.add_argument('--model',default='gpt-4o'); p.add_argument('--temperature',type=float,default=0.1)
    p.add_argument('--retries',type=int,default=3)
    a=p.parse_args(); k,v={'ntu60':(60,25),'ntu120':(120,25),'nwucla':(10,20)}[a.dataset]
    prompts=json.loads(Path(a.prompts).read_text(encoding='utf-8'))
    if not isinstance(prompts,list) or len(prompts)!=k or any(not isinstance(t,str) or not t.strip() for t in prompts): p.error(f'Provide exactly {k} complete prompts')
    out=Path(a.output)
    if out.suffix!='.npy' or out.exists(): p.error('Specify a new .npy output path')
    if a.retries<1: p.error('--retries must be positive')
    from openai import OpenAI  # optional dependency; OPENAI_API_KEY environment variable
    client=OpenAI(); results=[]
    for i,prompt in enumerate(prompts):
        for attempt in range(a.retries):
            try:
                response=client.chat.completions.create(model=a.model,temperature=a.temperature,messages=[{'role':'user','content':prompt}])
                content=response.choices[0].message.content.strip()
                if content.startswith('```'):
                    content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
                data=json.loads(content)
                results.append(validate_matrix(data,v)); print(f'Accepted class {i+1}/{k}',flush=True); break
            except (ValueError,TypeError,AttributeError) as e:
                if attempt==a.retries-1: raise RuntimeError(f'Class {i}: no valid matrix') from e
        # Accepted matrices are not symmetrized, clipped, or normalized here.
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f: np.save(f,np.stack(results),allow_pickle=False)
    print('Saved',out)

if __name__=='__main__': main()
