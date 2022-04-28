import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ..plan.rl_linear import AClinear, Qlinear
import numpy as np
from ..select_funcs import epsilon_select, policy_select
from functools import partial
from .grad_recognize import MyLSTM
from .donenet import DoneNet


class MyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_rcts(self):
        self.rct_dtypes = self.rec.rct_dtypes
        self.rct_shapes = self.rec.rct_shapes
        for k in self.rct_dtypes:
            setattr(self, k, getattr(self.rec, k))


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
        self.vobs_embed = nn.Linear(np.prod(obs_spc['fc'].shape), 512)
        self.tobs_embed = nn.Linear(np.prod(obs_spc['wd'].shape), 512)
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
        if q_flag:
            self.plan_out = Qlinear(i_size, act_spc.n)
            self.select_func = partial(epsilon_select, self.eps)
        else:
            self.plan_out = AClinear(i_size, act_spc.n)
            self.select_func = policy_select

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['wd']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.infer(x, (rct['hx'], rct['cx']))
        out = self.plan_out(h)
        out['action'] = self.select_func(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class BaseDoneModel(MyBase):
    """Baseline + DoneNet"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        done_net_path,
        done_thres,
        dropout_rate,
        learnable_x,
        init
    ):
        super().__init__()

        self.done_thres = done_thres
        self.done_idx = act_spc.n - 1
        fc_sz = np.prod(obs_spc['fc'].shape)
        wd_sz = np.prod(obs_spc['wd'].shape)
        self.vobs_embed = nn.Linear(fc_sz, 512)
        self.tobs_embed = nn.Linear(wd_sz, 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        i_size = 512
        self.rec = MyLSTM(1024, i_size, learnable_x, init)
        # plan
        self.DoneNet = DoneNet(fc_sz+wd_sz, 512)
        self.done_net.load_state_dict(torch.load(done_net_path))
        self.plan = AClinear(i_size, act_spc.n-1)
        self.get_rcts()

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['wd']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.rec(x, (rct['hx'], rct['cx']))
        # plan
        with torch.no_grad():
            done = self.done_net(torch.cat([obs['fc'], obs['wd']], dim=1))
        out = self.plan(h)
        action = policy_select(out).detach()
        action[done >= self.done_thres] = self.done_idx
        out['action'] = action
        out['rct'] = dict(hx=h, cx=c)
        return out