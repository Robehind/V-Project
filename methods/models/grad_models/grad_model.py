import torch
import torch.nn as nn
import numpy as np
from .grad_recognize import MyLSTM
from ..plan import AClinear
from ..select_funcs import policy_select
from .basemodel import MyBase
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class TgtAttModel(MyBase):
    """"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate,
        learnable_x: bool,
        init: str
    ):
        super().__init__()
        self.vobs_embed = nn.Linear(np.prod(obs_spc['fc'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        i_size = 512
        self.rec = MyLSTM(512, i_size, learnable_x, init)
        # plan
        self.plan_out = AClinear(i_size, act_spc.n)
        self.select_func = policy_select
        # target attention
        wd_sz = np.prod(obs_spc['wd'].shape)
        self.tgt_att = nn.Sequential(
            nn.Linear(wd_sz+i_size, 512),
            nn.ReLU(),
            nn.Linear(512, i_size),
            nn.Sigmoid())
        self.get_rcts()

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        h, c = self.rec(vobs_embed, (rct['hx'], rct['cx']))
        # x = self.drop(x)
        att_input = torch.cat([c, obs['wd']], dim=1)
        tgt_att = self.tgt_att(att_input)
        out = self.plan_out(h*tgt_att)
        out['action'] = self.select_func(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class ActMatModel(MyBase):
    """"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate,
        learnable_x: bool,
        init: str
    ):
        super().__init__()
        # 动作变换矩阵
        fc_sz = np.prod(obs_spc['fc'].shape)
        wd_sz = np.prod(obs_spc['wd'].shape)
        self.vobs_embed = nn.Linear(fc_sz, 512)
        self.tobs_embed = nn.Linear(wd_sz, 512)
        i_size = 512
        self.act_linear0 = nn.Linear(i_size, i_size, bias=False)
        self.act_linear1 = nn.Linear(i_size, i_size, bias=False)
        # 向右转的处理可能可以是对应的，比如简单的取负？
        self.act_linear2 = nn.Linear(i_size, i_size, bias=False)
        self.drop = nn.Dropout(p=dropout_rate)
        # LSTM
        self.rec = MyLSTM(fc_sz+wd_sz, i_size, learnable_x, init)
        # plan
        self.plan = AClinear(i_size, act_spc.n)
        self.get_rcts()
        self.rct_dtypes.update(action=torch.int64)
        self.rct_shapes.update(action=(1,))
        self.action = Parameter(
            3*torch.ones((1,), dtype=torch.int64), False)

    def forward(self, obs, rct):
        idx = rct['action']
        hx, cx = rct['hx'], rct['cx']
        # 前进、左转、右转将使It发生某种转换
        new_hx = []
        for i in range(hx.shape[0]):
            tmp = hx[i]
            if idx[i] <= 2:
                tmp = getattr(self, 'act_linear'+str(idx[i].item()))(hx[i])
            new_hx.append(tmp)
        n_hx = torch.stack(new_hx, dim=0)
        x = torch.cat([obs['fc'], obs['wd']], dim=1)
        h, c = self.rec(x, (n_hx, cx))
        out = self.plan(h)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        return out


class TgtAttActMatModel(MyBase):
    """"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        learnable_x: bool,
        init: str
    ):
        super().__init__()
        # 动作变换矩阵
        fc_sz = np.prod(obs_spc['fc'].shape)
        wd_sz = np.prod(obs_spc['wd'].shape)
        i_size = 512
        self.mMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        self.rMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        # weight init
        stdv = 1. / math.sqrt(i_size)
        self.mMat.data.uniform_(-stdv, stdv)
        self.rMat.data.uniform_(-stdv, stdv)
        # LSTM
        self.rec = MyLSTM(fc_sz, i_size, learnable_x, init)
        # target attention
        self.tgt_att = nn.Sequential(
            nn.Linear(wd_sz, 512),
            nn.ReLU(),
            nn.Linear(512, i_size),
            nn.Sigmoid())
        # plan
        self.plan = AClinear(i_size, act_spc.n)
        self.get_rcts()
        self.rct_dtypes.update(action=torch.int64)
        self.rct_shapes.update(action=(1,))
        self.action = Parameter(
            3*torch.ones((1,), dtype=torch.int64), False)

    def forward(self, obs, rct):
        idx = rct['action']
        hx, cx = rct['hx'], rct['cx']
        # 前进、左转、右转将使It发生某种转换，碰撞了就不变换
        new_hx = []
        for i in range(hx.shape[0]):
            tmp = hx[i]
            if idx[i] == 0 and not obs['collision'][i]:
                tmp = torch.matmul(hx[i], self.mMat)
            if idx[i] == 1:
                tmp = torch.matmul(hx[i], self.rMat)
            if idx[i] == 2:
                tmp = torch.matmul(hx[i], self.rMat.T)
            new_hx.append(tmp)
        n_hx = torch.stack(new_hx, dim=0)
        h, c = self.rec(obs['fc'], (n_hx, cx))
        # plan
        tgt_att = self.tgt_att(obs['wd'])
        out = self.plan(h*tgt_att)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        return out
