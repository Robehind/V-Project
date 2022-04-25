import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .grad_recognize import MyLSTM
from .grad_plan import SplitDone
from .grad_perception import MyCNN
from ..plan import AClinear
from ..select_funcs import policy_select
from .basemodel import MyBase


class CNNmodel(MyBase):
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate,
        learnable_x: bool,
        init: str
    ):
        super().__init__()
        self.img_per = MyCNN()
        cnn_sz = self.img_per.out_fc_sz(*obs_spc['frame'].shape[:2])
        self.vobs_embed = nn.Linear(cnn_sz, 512)
        self.tobs_embed = nn.Linear(np.prod(obs_spc['wd'].shape), 512)
        i_sz = 512
        self.rec = MyLSTM(512+512, i_sz, learnable_x, init)
        self.plan = AClinear(i_sz, act_spc.n)
        self.drop = nn.Dropout(p=dropout_rate)
        self.get_rcts()

    def forward(self, obs, rct):
        frames = obs['frame'].permute(0, 3, 1, 2) / 255.
        cnn_tmp = F.relu(self.img_per(frames))
        cnn_out = torch.flatten(cnn_tmp, 1)
        vobs_embed = F.relu(self.vobs_embed(cnn_out), True)
        tobs_embed = F.relu(self.tobs_embed(obs['wd']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.rec(x, (rct['hx'], rct['cx']))
        out = self.plan(h)
        out['rct'] = dict(hx=h, cx=c)
        out['action'] = policy_select(out).detach()
        return out


class SplitDoneModel(MyBase):
    """单独的done网络"""
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
        self.tobs_embed = nn.Linear(np.prod(obs_spc['wd'].shape), 512)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        i_size = 512
        self.rec = MyLSTM(1024, i_size, learnable_x, init)
        # plan
        self.plan = SplitDone(i_size, act_spc.n)
        # Done Plan
        self.done_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        self.get_rcts()

    def forward(self, obs, rct):
        vobs_embed = F.relu(self.vobs_embed(obs['fc']), True)
        tobs_embed = F.relu(self.tobs_embed(obs['wd']), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        h, c = self.rec(x, (rct['hx'], rct['cx']))
        done_pre = self.done_net(x)
        out = self.plan(h, done_pre)
        out['rct'] = dict(hx=h, cx=c)
        return out
