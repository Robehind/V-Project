import torch.nn.functional as F
import torch.nn as nn
import torch
from ..select_funcs import policy_select
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
        out['action'] = policy_select(out).detach()
        return out


class Zhu2017Act(nn.Module):
    """Zhu et al. Target-driven visual navigation in Indoor scenes \
        using Deep Reinforcement Learning with choosen act"""
    def __init__(
        self,
        obs_spc: Dictspc,
        act_spc: Discrete,
        act_embed_sz: int,
        infer_sz=512
    ):
        super().__init__()
        self.act_l = nn.Linear(1, act_embed_sz)
        self.p = SiameseLinear(np.prod(obs_spc['fc'].shape), 512)
        self.m = nn.LSTMCell(1024+act_embed_sz, infer_sz)
        act_sz = act_spc.n
        self.d = AClinear(infer_sz, act_sz)
        self.rct_shapes = {
            'hx': (infer_sz,), 'cx': (infer_sz,), 'action': (1,)}
        dtype = next(self.m.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype, 'action': torch.int64}

    def forward(self, obs, rct):
        x = F.relu(self.p(obs['fc'], obs['tgt']), True)
        act = F.relu(self.act_l(rct['action'].float()), True)
        x = torch.cat((x, act), dim=1)
        hx, cx = rct['hx'], rct['cx']
        nh, nc = self.m(x, (hx, cx))
        out = self.d(nh)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=nh, cx=nc, action=out['action'].unsqueeze(1))
        return out


class Zhu2017RepAct(nn.Module):
    """Zhu et al. Target-driven visual navigation in Indoor scenes \
        using Deep Reinforcement Learning with choosen act"""
    def __init__(
        self,
        obs_spc: Dictspc,
        act_spc: Discrete,
        act_rep_sz: int,
        infer_sz=512
    ):
        super().__init__()
        # self.act_l = nn.Linear(1, act_embed_sz)
        self.act_rep_sz = act_rep_sz
        self.p = SiameseLinear(np.prod(obs_spc['fc'].shape), 512)
        self.m = nn.LSTMCell(1024+act_rep_sz, infer_sz)
        act_sz = act_spc.n
        self.d = AClinear(infer_sz, act_sz)
        self.rct_shapes = {
            'hx': (infer_sz,), 'cx': (infer_sz,), 'action': (1,)}
        dtype = next(self.m.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype, 'action': torch.int64}

    def forward(self, obs, rct):
        x = F.relu(self.p(obs['fc'], obs['tgt']), True)
        act = rct['action'].repeat(1, self.act_rep_sz)
        x = torch.cat((x, act), dim=1)
        hx, cx = rct['hx'], rct['cx']
        nh, nc = self.m(x, (hx, cx))
        out = self.d(nh)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=nh, cx=nc, action=out['action'].unsqueeze(1))
        return out
