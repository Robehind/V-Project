import torch
import torch.nn as nn
import torch.nn.functional as F
from ..plan.rl_linear import AClinear
import numpy as np
from ..select_funcs import policy_select
from .grad_recognize import MyLSTM
from gym.spaces import Discrete
from .donenet import DoneNet


class MyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_rcts(self):
        self.rct_dtypes = self.rec.rct_dtypes
        self.rct_shapes = self.rec.rct_shapes
        for k in self.rct_dtypes:
            setattr(self, k, getattr(self.rec, k))


class BaseLstmModel(MyBase):
    """观察都是预处理好的特征向量的lstm模型,上一时刻的动作也作为其输入"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate,
        learnable_x,
        init
    ):
        super().__init__()
        self.vobs_embed = nn.Linear(np.prod(obs_spc['fc'].shape), 512)
        self.tobs_embed = nn.Linear(np.prod(obs_spc['wd'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        i_size = 512
        self.rec = MyLSTM(1024, i_size, learnable_x, init)
        # plan
        self.plan = AClinear(i_size, act_spc.n)
        self.get_rcts()

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['wd']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.rec(x, (rct['hx'], rct['cx']))
        out = self.plan(h)
        out['action'] = policy_select(out).detach()
        out['rct'] = dict(hx=h, cx=c)
        return out


class BaseDoneModel(BaseLstmModel):
    """Baseline + DoneNet"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate,
        learnable_x,
        init,
        done_net_path,
        done_thres,
    ):
        _act_spc = Discrete(act_spc.n-1)
        super().__init__(obs_spc, _act_spc, dropout_rate,
                         learnable_x, init)
        self.done_thres = done_thres
        self.done_idx = act_spc.n - 1
        fc_sz = np.prod(obs_spc['fc'].shape)
        wd_sz = np.prod(obs_spc['wd'].shape)
        self.done_net = DoneNet(fc_sz+wd_sz, 512, 0)
        self.done_net.load_state_dict(torch.load(done_net_path))
        self.done_net.eval()
        self.big_num = 9999999

    def forward(self, obs, rct):
        out = super().forward(obs, rct)
        action = out['action'].unsqueeze(1)
        # done plan
        with torch.no_grad():
            done = self.done_net(torch.cat([obs['fc'], obs['wd']], dim=1))
        idx = done >= self.done_thres
        action[idx] = self.done_idx
        # 加一行policy凑够动作，done是不带梯度的
        inf_pre = -torch.ones_like(done)
        inf_pre[idx] = -inf_pre[idx]
        out['policy'] = torch.cat([out['policy'], self.big_num*inf_pre], dim=1)
        out['action'] = action.squeeze()
        return out
