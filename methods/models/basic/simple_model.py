import torch
import torch.nn as nn
import torch.nn.functional as F
from ..plan.rl_linear import AClinear, Qlinear
import numpy as np
from ..select_funcs import epsilon_select, policy_select
from functools import partial


class SimpleMP(torch.nn.Module):
    """vobs和tobs都是已经被处理好的特征向量,没有封装input，不能直接用于训练"""
    def __init__(
        self,
        action_sz,
        vobs_sz,
        tobs_sz,
        tobs_embed_sz=512,
        vobs_embed_sz=512,
        infer_sz=512,
        dropout_rate=0,
        mode=0,  # 0 for linear 1 for lstm
        q_flag=0,
        eps=0.1
    ):
        super(SimpleMP, self).__init__()
        self.eps = eps
        self.vobs_embed = nn.Linear(vobs_sz, vobs_embed_sz)
        self.tobs_embed = nn.Linear(tobs_sz, tobs_embed_sz)
        # mem&infer
        self.drop = nn.Dropout(p=dropout_rate)
        self.mode = mode
        if mode == 0:
            self.infer = nn.Sequential(
                nn.Linear(tobs_embed_sz+vobs_embed_sz, infer_sz),
                nn.ReLU(True),
                nn.Linear(infer_sz, infer_sz)
            )
        else:
            self.infer = nn.LSTMCell(tobs_embed_sz+vobs_embed_sz, infer_sz)
            self.rct_shapes = {'hx': (infer_sz, ), 'cx': (infer_sz, )}
            dtype = next(self.infer.parameters()).dtype
            self.rct_dtypes = {'hx': dtype, 'cx': dtype}
        # plan
        if q_flag:
            self.plan_out = Qlinear(infer_sz, action_sz)
            self.select_func = partial(epsilon_select, self.eps)
        else:
            self.plan_out = AClinear(infer_sz, action_sz)
            self.select_func = policy_select
        # self.apply(weights_init)

    def forward(self, vobs, tobs, hx, cx):

        vobs_embed = F.relu(self.vobs_embed(vobs), True)
        tobs_embed = F.relu(self.tobs_embed(tobs), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        if self.mode == 0:
            x = self.infer(x)
            out = self.plan_out(x)
            out['action'] = self.select_func(out)
            return out
        h, c = self.infer(x, (hx, cx))
        out = self.plan_out(h)
        n_rct = dict(hx=h, cx=c)
        out['rct'] = n_rct
        out['action'] = self.select_func(out)
        return out


class FcLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        dropout_rate=0,
        q_flag=0
    ):
        super(FcLstmModel, self).__init__()
        self.net = SimpleMP(act_spc.n,
                            np.prod(obs_spc['fc'].shape),
                            np.prod(obs_spc['glove'].shape),
                            dropout_rate=dropout_rate,
                            mode=1, q_flag=q_flag)
        self.rct_shapes = self.net.rct_shapes
        self.rct_dtypes = self.net.rct_dtypes

    def forward(self, obs, rct):
        return self.net(
            obs['fc'],
            obs['glove'],
            rct['hx'],
            rct['cx'])


class FcLinearModel(torch.nn.Module):
    """观察都是预处理好的特征向量的linear模型"""
    def __init__(
        self,
        obs_shapes,
        act_sz,
        dropout_rate=0,
        q_flag=0
    ):
        super(FcLinearModel, self).__init__()
        self.vobs_sz = np.prod(obs_shapes['fc'])
        tobs_sz = np.prod(obs_shapes['glove'])

        self.net = SimpleMP(act_sz, self.vobs_sz, tobs_sz,
                            dropout_rate=dropout_rate, q_flag=q_flag)

    def forward(self, obs, rct={}):

        vobs_embed = torch.flatten(obs['fc']) \
            .view(-1, self.vobs_sz)
        return self.net(vobs_embed, obs['glove'], None, None)


class FcActLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型,上一时刻的动作也作为其输入"""
    def __init__(
        self,
        obs_spc,
        act_spc,
        act_embed_sz=10,
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
