import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ..plan.rl_linear import AClinear, Qlinear
import numpy as np
from ..select_funcs import epsilon_select, policy_select
from functools import partial


class BaseLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型,上一时刻的动作也作为其输入"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate=0,
        q_flag=0,
        learnable_x=False
    ):
        super().__init__()
        self.vobs_embed = nn.Linear(
            np.prod(obs_spc['fc'].shape), 512)
        self.tobs_embed = nn.Linear(
            np.prod(obs_spc['glove'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        self.mode = q_flag
        i_size = 512
        self.infer = nn.LSTMCell(1024, i_size)
        self.rct_shapes = {'hx': (i_size, ), 'cx': (i_size, )}
        dtype = next(self.infer.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype}
        if learnable_x:
            self.hx = Parameter(torch.randn(1, i_size))
            self.cx = Parameter(torch.randn(1, i_size))
        # plan
        if q_flag:
            self.plan_out = Qlinear(i_size, act_spc.n)
            self.select_func = partial(epsilon_select, self.eps)
        else:
            self.plan_out = AClinear(i_size, act_spc.n)
            self.select_func = policy_select

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['glove']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.infer(x, (rct['hx'], rct['cx']))
        out = self.plan_out(h)
        out['action'] = self.select_func(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class BaseActLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型,上一时刻的动作也作为其输入"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        act_embed_sz=64,
        dropout_rate=0,
        q_flag=0
    ):
        super().__init__()
        self.act_embed = nn.Linear(1, act_embed_sz)
        self.vobs_embed = nn.Linear(
            np.prod(obs_spc['fc'].shape), 512)
        self.tobs_embed = nn.Linear(
            np.prod(obs_spc['glove'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        self.mode = q_flag
        i_size = 512
        self.infer = nn.LSTMCell(1024+act_embed_sz, i_size)
        self.rct_shapes = {
            'hx': (i_size, ), 'cx': (i_size, ), 'action': (1,)}
        dtype = next(self.infer.parameters()).dtype
        self.rct_dtypes = {
            'hx': dtype, 'cx': dtype, 'action': torch.int64}
        # plan
        if q_flag:
            self.plan_out = Qlinear(i_size, act_spc.n)
            self.select_func = partial(epsilon_select, self.eps)
        else:
            self.plan_out = AClinear(i_size, act_spc.n)
            self.select_func = policy_select

    def forward(self, obs, rct):
        act = F.relu(self.act_embed(rct['action'].float()), True)
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['glove']), True)
        x = torch.cat((vobs_embed, tobs_embed, act), dim=1)
        x = self.drop(x)
        h, c = self.infer(x, (rct['hx'], rct['cx']))
        out = self.plan_out(h)
        out['action'] = self.select_func(out).detach()
        n_rct = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        out['rct'] = n_rct
        return out
