"""Prior-tool tests: mocked API only, no SDK or network required."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.generate_priors import load_prompt_config,generate_bank,validate_matrix,validate_output


class PriorGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.raw=json.loads(Path('priors/prompts/ntu60.json').read_text(encoding='utf-8'))
        self.hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('priors').glob('*.npy')}
    def tearDown(self):
        for p,h in self.hashes.items(): self.assertEqual(h,hashlib.sha256(p.read_bytes()).hexdigest())
    def write(self,value):
        p=self.root/'prompts.json'; p.write_text(json.dumps(value),encoding='utf-8'); return p
    def test_original_and_request(self):
        s=load_prompt_config(self.write(self.raw),'ntu60')
        self.assertEqual(len(s['prompts']),60); self.assertEqual(len(self.raw['joint_names']),25)
        self.assertEqual(self.raw['joint_names'][20],'spine')
        self.assertEqual(self.raw['class_names'][8],'standing up')
        self.assertEqual(self.raw['class_names'][25],'hopping (one foot)')
        self.assertEqual(self.raw['class_names'][26],'jumping up')
        matrix=np.eye(25,dtype=np.float32); matrix[0,1]=.2
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(matrix.tolist())))])))))
        out=generate_bank(client,s,self.root/'new.npy')
        a=np.load(out); self.assertEqual(a.shape,(60,25,25)); self.assertEqual(a.dtype,np.float32)
        self.assertTrue(np.array_equal(a[0],matrix)); self.assertEqual(a[0,1,0],0)
        calls=client.chat.completions.create.call_args_list
        self.assertEqual(len(calls),60)
        for i,call in enumerate(calls):
            self.assertEqual(set(call[1]),{'model','temperature','messages'})
            self.assertEqual(call[1]['model'],'gpt-4o'); self.assertEqual(call[1]['temperature'],.1)
            self.assertEqual(call[1]['messages'],[dict(role='system',content='You are a helpful assistant that strictly outputs raw JSON 2D arrays.'),dict(role='user',content=self.raw['prompts'][i])])
        with self.assertRaises(ValueError): generate_bank(client,s,out)
        with self.assertRaises(ValueError): validate_output('priors/ntu60.npy')
    def test_legacy_and_priority(self):
        s=load_prompt_config(self.write(self.raw['prompts']),'ntu60')
        self.assertIsNone(s['system_prompt'])
        s=load_prompt_config(self.write(self.raw),'ntu60','custom-model',.3)
        self.assertEqual((s['model'],s['temperature']),('custom-model',.3))
        from tools.generate_priors import build_messages
        old=load_prompt_config(self.write(self.raw['prompts']),'ntu60')
        self.assertEqual(build_messages(old,old['prompts'][0]),[dict(role='user',content=self.raw['prompts'][0])])
    def test_invalid_configs(self):
        for patch in (dict(dataset='ntu120'),dict(prompts=[]),dict(prompts=['']*60),dict(system_prompt=1),dict(class_names=['x']),dict(joint_names=['x']),dict(temperature=float('nan')),dict(temperature=3),dict(model=''),dict(prompts=['{action}']*60)):
            raw=copy.deepcopy(self.raw); raw.update(patch)
            with self.assertRaises((ValueError,TypeError,KeyError)): load_prompt_config(self.write(raw),'ntu60')
    def test_invalid_matrices_and_retry(self):
        for a in (np.eye(24),np.full((25,25),np.nan),np.full((25,25),np.inf),np.ones((25,25))*1.1,np.zeros((25,25))):
            with self.assertRaises(ValueError): validate_matrix(a,25)
        s=load_prompt_config(self.write(self.raw),'ntu60'); client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[]'))])))))
        with self.assertRaises(RuntimeError): generate_bank(client,s,self.root/'invalid.npy',2)
        self.assertEqual(client.chat.completions.create.call_count,2); self.assertFalse((self.root/'invalid.npy').exists())

if __name__=='__main__': unittest.main()
