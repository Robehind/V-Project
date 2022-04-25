import torch
import torch.nn as nn
import numpy as np
from .grad_recognize import MyLSTM
from .grad_plan import SplitDone
from .basemodel import MyBase
from torch.nn.parameter import Parameter


class GradModel(MyBase):
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
        i_size = 1024
        self.act_linear0 = nn.Linear(i_size, i_size, bias=False)
        self.act_linear1 = nn.Linear(i_size, i_size, bias=False)
        # 向右转的处理可能可以是对应的，比如简单的取负？
        self.act_linear2 = nn.Linear(i_size, i_size, bias=False)
        self.drop = nn.Dropout(p=dropout_rate)
        # LSTM
        self.rec = MyLSTM(fc_sz, i_size, learnable_x, init)
        # plan
        self.plan = SplitDone(i_size, act_spc.n)
        # target attention
        self.tgt_att = nn.Sequential(
            nn.Linear(wd_sz, i_size),
            # nn.ReLU(),
            # nn.Linear(i_size, i_size),
            nn.Sigmoid())
        # Done Plan
        self.done_net = nn.Sequential(
            nn.Linear(fc_sz + wd_sz, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
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
        h, c = self.rec(obs['fc'], (n_hx, cx))
        # 单独的done网络
        done_feat = torch.cat([obs['fc'], obs['wd']], dim=1)
        done_pre = self.done_net(done_feat)
        # 目标作为It上的一种注意力
        tgt_att = self.tgt_att(obs['wd'])
        out = self.plan(h*tgt_att, done_pre)
        out['rct'] = dict(hx=h, cx=c, action=out['action'].unsqueeze(1))
        return out
