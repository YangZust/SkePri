"""PYSKL-format NTU annotations; preprocessing order follows source configs."""
import pickle
import numpy as np
from torch.utils.data import Dataset
from .tools import PreNormalize3D, JointToBone, ToMotion, UniformSampleFrames, PoseDecode, FormatGCNInput

class Feeder(Dataset):
    def __init__(self, path, protocol, train, stream, clip_len=100, num_clips=1, repeat=1):
        with open(path,'rb') as f:
            data=pickle.load(f)  # only load trusted annotations
        split=protocol+('_train' if train else '_val')
        ids=set(data['split'][split])
        self.items=[a for a in data['annotations'] if a['frame_dir'] in ids]
        if len(self.items)!=len(ids) or not self.items:
            raise ValueError(f'Invalid or empty split {split}')
        self.train, self.stream, self.repeat=train, stream, repeat
        self.sample=UniformSampleFrames(clip_len,num_clips=num_clips)
        self.normalize=PreNormalize3D()
        self.format=FormatGCNInput(num_person=2)
    def __len__(self): return len(self.items)*self.repeat
    def __getitem__(self,index):
        a=self.items[index%len(self.items)]
        d=dict(a, keypoint=a['keypoint'].copy(), start_index=0, test_mode=not self.train)
        d=self.normalize(d)
        if self.stream in ('bone','bone_motion'):
            d=JointToBone(target='keypoint')(d)
        if self.stream in ('joint_motion','bone_motion'):
            d=ToMotion(target='keypoint')(d)
        d=self.format(PoseDecode()(self.sample(d)))
        return d['keypoint'].astype(np.float32), int(a['label']), a['frame_dir']
