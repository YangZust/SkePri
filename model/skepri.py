"""SkePri: integrated skeleton model. NTU layers adapted from PYSKL, UCLA layers
from CTR-GCN. Original notices/licenses: licenses/. SkePri auxiliary branch
implements Eqs. (4)-(12) of the supplied manuscript.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph.ntu_rgb_d import Graph as NTUGraph
from graph.ucla import Graph as UCLAGraph

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        rel = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        out = torch.einsum('ncuv,nctu->nctv', rel, x3)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)

class MSTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 act_cfg=dict(type='ReLU'),
                 tcn_dropout=0):

        super().__init__()
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (self.num_branches - 1)

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(),
                unit_tcn(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels_rem, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels_rem)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                bn_init(m, 1)

class CTRGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2],
                 tcn_dropout=0):
        super(CTRGCNBlock, self).__init__()
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn1 = MSTCN(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
            tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

    def init_weights(self):
        self.tcn1.init_weights()
        self.gcn1.init_weights()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class UCLARelation(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(UCLARelation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class UCLATemporalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(UCLATemporalUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class UCLAGraphUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(UCLAGraphUnit, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(UCLARelation(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y

class UCLABlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(UCLABlock, self).__init__()
        self.gcn1 = UCLAGraphUnit(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UCLATemporalUnit(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    """Input [N,M,T,V,C]; output main logits, plus auxiliary logits in training.

    With a prior: temporal pooling then person averaging defines G [N,D,V].
    Person averaging precedes the nonlinear pairwise interaction (explicit
    implementation convention for the manuscript's person-free notation).
    """
    def __init__(self, dataset, num_classes, num_person, prior_path=None,
                 aux_weight=0.01, dropout=None, prior_chunk=16):
        super().__init__()
        if dataset not in ('ntu60','ntu120','nwucla'):
            raise ValueError(f'Unsupported dataset: {dataset}')
        expected = {'ntu60': (60,2), 'ntu120': (120,2), 'nwucla': (10,1)}[dataset]
        if (num_classes,num_person) != expected:
            raise ValueError('Dataset/class/person mismatch')
        self.dataset, self.num_classes, self.num_person = dataset, num_classes, num_person
        self.num_point = 20 if dataset == 'nwucla' else 25
        self.aux_weight = float(aux_weight)
        self.prior_chunk = int(prior_chunk)
        if self.prior_chunk < 1 or self.aux_weight < 0:
            raise ValueError('Invalid auxiliary configuration')
        ucla = dataset == 'nwucla'
        graph = UCLAGraph() if ucla else NTUGraph(layout='nturgb+d', mode='spatial')
        A = graph.A if ucla else torch.tensor(graph.A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(num_person * 3 * self.num_point)
        self.blocks = nn.ModuleList()
        in_channels = 3
        for i in range(10):
            channels = 64 if i < 4 else 128 if i < 7 else 256
            stride = 2 if i in (4,7) else 1
            if ucla:
                layer = UCLABlock(in_channels, channels, A, stride=stride, residual=i!=0)
            else:
                layer = CTRGCNBlock(in_channels, channels, A.clone(), stride=stride, residual=i!=0)
                layer.init_weights()
            self.blocks.append(layer)
            in_channels = channels
        self.dropout = nn.Dropout(0.0 if dropout is None else dropout)
        self.classifier = nn.Linear(256, num_classes)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2./num_classes) if ucla else 0.01)
        if not ucla:
            nn.init.zeros_(self.classifier.bias)
        bn_init(self.data_bn, 1)
        self.auxiliary = None
        if prior_path is not None:
            a = np.load(prior_path, allow_pickle=False)
            if a.shape != (num_classes,self.num_point,self.num_point):
                raise ValueError(f'Invalid prior shape: {a.shape}')
            if not np.isfinite(a).all() or a.min()<0 or a.max()>1 or not np.allclose(np.diagonal(a,axis1=-2,axis2=-1),1,atol=1e-7,rtol=0):
                raise ValueError('Prior must be finite in [0,1] with unit diagonal')
            a = torch.tensor(a, dtype=torch.float32)
            self.register_buffer('prior', a / (a.sum(-1,keepdim=True)+1e-6))
            self.auxiliary = nn.Linear(256,1)
            nn.init.normal_(self.auxiliary.weight, 0, 0.01)
            nn.init.zeros_(self.auxiliary.bias)

    def features(self, x):
        n,m,t,v,c = x.shape
        if (m,v,c) != (self.num_person,self.num_point,3):
            raise ValueError(f'Expected [N,{self.num_person},T,{self.num_point},3], got {tuple(x.shape)}')
        x = x.permute(0,1,3,4,2).contiguous().reshape(n,m*v*c,t)
        x = self.data_bn(x).reshape(n,m,v,c,t).permute(0,1,3,4,2).contiguous().reshape(n*m,c,t,v)
        for layer in self.blocks:
            x = layer(x)
        return x.reshape(n,m,256,x.shape[2],v)

    def auxiliary_scores(self, features):
        # Eq. (4): temporal mean; average the person axis for sample-level G.
        g = features.mean(dim=3).mean(dim=1)
        # Eqs. (6)-(8): G A, element-wise product with G, node mean, shared D->1.
        # Class chunks bound intermediate memory; no class-specific classifier.
        scores = []
        for a in self.prior.split(self.prior_chunk, dim=0):
            z = torch.einsum('ndv,kvw->nkdw', g, a)
            pooled = (z * g.unsqueeze(1)).mean(dim=-1)
            scores.append(self.auxiliary(pooled).squeeze(-1))
        return torch.cat(scores, dim=1)

    def forward(self, x):
        f = self.features(x)
        main = self.classifier(self.dropout(f.mean(dim=(1,3,4))))
        if self.training and self.auxiliary is not None:
            return main, self.auxiliary_scores(f)
        return main

    def loss(self, x, y):
        out = self(x)
        if isinstance(out, tuple):
            main, aux = out
            return F.cross_entropy(main,y) + self.aux_weight * F.cross_entropy(aux,y), main
        return F.cross_entropy(out,y), out

    def inference_state_dict(self):
        """Export without prior buffer or train-only parameters."""
        return {k:v for k,v in self.state_dict().items() if k!='prior' and not k.startswith('auxiliary.')}
