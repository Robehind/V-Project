import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ..plan.rl_linear import AClinear
import numpy as np
from ..select_funcs import policy_select


class SplitDoneModel(torch.nn.Module):
    """单独的done网络"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate=0,
        q_flag=0,
        learnable_x=False
    ):
        super().__init__()
        self.vobs_embed = nn.Linear(np.prod(obs_spc['fc'].shape), 512)
        self.tobs_embed = nn.Linear(np.prod(obs_spc['glove'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        self.mode = q_flag
        i_size = 512
        self.infer = nn.LSTMCell(1024, i_size)
        self.rct_shapes = {'hx': (i_size, ), 'cx': (i_size, )}
        dtype = next(self.infer.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype}
        self.hx = Parameter(torch.zeros(1, i_size), learnable_x)
        self.cx = Parameter(torch.zeros(1, i_size), learnable_x)
        # plan
        self.plan_out = AClinear(i_size, act_spc.n - 1)
        self.select_func = policy_select
        # Done Plan
        self.done_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['glove']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.infer(x, (rct['hx'], rct['cx']))
        out = self.plan_out(h)
        done_pre = self.done_net(x)
        # 一定要保证done在最后一个
        out['policy'] = torch.cat([out['policy'], done_pre], dim=1)
        out['action'] = self.select_func(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out
