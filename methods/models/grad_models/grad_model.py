import torch
import torch.nn as nn
import numpy as np
from .grad_recognize import MyLSTM
from ..plan import AClinear
from ..select_funcs import policy_select
from .basemodel import MyBase
from torch.nn.parameter import Parameter
import math
from .donenet import done_wrapper


class TgtAttModel(MyBase):
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

    def forward(self, obs, rct):
        hx, cx = rct['hx'], rct['cx']
        h, c = self.rec(obs['fc'], (hx, cx))
        # plan
        tgt_att = self.tgt_att(obs['wd'])
        out = self.plan(h*tgt_att)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class ActMatModel(MyBase):
    """"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        learnable_x: bool,
        init: str
    ):
        super().__init__()
        # embedding
        fc_sz = np.prod(obs_spc['fc'].shape)
        wd_sz = np.prod(obs_spc['wd'].shape)
        self.vobs_embed = nn.Sequential(
            nn.Linear(fc_sz, 512), nn.ReLU())
        self.tobs_embed = nn.Sequential(
            nn.Linear(wd_sz, 512), nn.ReLU())
        # 动作变换矩阵
        i_size = 512
        self.mMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        self.rMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        # self.lMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        # weight init
        stdv = 1. / math.sqrt(i_size)
        self.mMat.data.uniform_(-stdv, stdv)
        self.rMat.data.uniform_(-stdv, stdv)
        # self.lMat.data.uniform_(-stdv, stdv)
        # LSTM
        self.rec = MyLSTM(1024, i_size, learnable_x, init)
        # plan
        self.plan = AClinear(i_size, act_spc.n)
        self.get_rcts()
        self.rct_dtypes.update(action=torch.int64)
        self.rct_shapes.update(action=(1,))
        self.action = Parameter(
            3*torch.ones((1,), dtype=torch.int64), False)

    def forward(self, obs, rct):
        vobs_embed = self.vobs_embed(obs['fc'])
        tobs_embed = self.tobs_embed(obs['wd'])
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
        h, c = self.rec(
            torch.cat([vobs_embed, tobs_embed], dim=1), (n_hx, cx))
        # plan
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
        self.lMat = Parameter(torch.FloatTensor(i_size, i_size), True)
        # weight init
        stdv = 1. / math.sqrt(i_size)
        self.mMat.data.uniform_(-stdv, stdv)
        self.rMat.data.uniform_(-stdv, stdv)
        self.lMat.data.uniform_(-stdv, stdv)
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
                tmp = torch.matmul(hx[i], self.lMat)
            new_hx.append(tmp)
        n_hx = torch.stack(new_hx, dim=0)
        h, c = self.rec(obs['fc'], (n_hx, cx))
        # plan
        tgt_att = self.tgt_att(obs['wd'])
        out = self.plan(h*tgt_att)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        return out


TgtAttDmodel = done_wrapper('TgtAttDoneModel', TgtAttModel)
ActMatDmodel = done_wrapper('ActMatDmodel', ActMatModel)
TgtAttActMatDmodel = done_wrapper('TgtAttActMatDmodel', TgtAttActMatModel)
