import torch.nn.functional as F
import torch.nn as nn
from ..perception.siamese_cnn import SiameseLinear
from ..plan.rl_linear import AClinear
import numpy as np
from gym.spaces import Dict as Dictspc
from gym.spaces import Discrete


class Zhu2017(nn.Module):
    """Zhu et al. Target-driven visual navigation in Indoor scenes \
        using Deep Reinforcement Learning"""
    def __init__(
        self,
        obs_spc: Dictspc,
        act_spc: Discrete,
        infer_sz=512
    ):
        super().__init__()
        self.p = SiameseLinear(np.prod(obs_spc['fc'].shape), 512)
        self.m = nn.LSTMCell(1024, infer_sz)
        act_sz = act_spc.n
        self.d = AClinear(infer_sz, act_sz)
        self.rct_shapes = {'hx': (infer_sz, ), 'cx': (infer_sz, )}
        dtype = next(self.m.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype}

    def forward(self, obs, rct):
        x = F.relu(self.p(obs['fc'], obs['tgt']), True)
        hx, cx = rct['hx'], rct['cx']
        nh, nc = self.m(x, (hx, cx))
        out = self.d(nh)
        out['rct'] = dict(hx=nh, cx=nc)
        return out
