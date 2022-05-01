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
        self.vobs_embed = nn.Sequential(
            nn.Linear(fc_sz, 512), nn.ReLU())
        i_size = 512
        # LSTM
        self.rec = MyLSTM(i_size, i_size, learnable_x, init)
        # target attention
        self.tgt_att = nn.Sequential(
            nn.Linear(wd_sz+i_size, 512),
            nn.ReLU(),
            nn.Linear(512, i_size),
            nn.Sigmoid())
        # plan
        self.plan = AClinear(i_size, act_spc.n)
        self.get_rcts()

    def forward(self, obs, rct):
        v_feat = self.vobs_embed(obs['fc'])
        hx, cx = rct['hx'], rct['cx']
        h, c = self.rec(v_feat, (hx, cx))
        # plan
        tgt_att = self.tgt_att(torch.cat([c, obs['wd']], dim=1))
        out = self.plan(h*tgt_att)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class ActVecModel(MyBase):
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
        # 动作变换向量
        i_size = 512
        self.Mat0 = Parameter(torch.FloatTensor(i_size), True)
        self.Mat1 = Parameter(torch.FloatTensor(i_size), True)
        self.Mat2 = Parameter(torch.FloatTensor(i_size), True)
        # weight init
        stdv = 1. / math.sqrt(i_size)
        self.Mat0.data.uniform_(-stdv, stdv)
        self.Mat1.data.uniform_(-stdv, stdv)
        self.Mat2.data.uniform_(-stdv, stdv)
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
        for i in range(hx.shape[0]):
            act_n = idx[i].item()
            if act_n <= 2 and not obs['collision'][i]:
                hx[i] *= getattr(self, 'Mat'+str(act_n))
        h, c = self.rec(
            torch.cat([vobs_embed, tobs_embed], dim=1), (hx, cx))
        # plan
        out = self.plan(h)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        return out


TgtAttDmodel = done_wrapper('TgtAttDoneModel', TgtAttModel)
ActVecDmodel = done_wrapper('ActVecDmodel', ActVecModel)
