import torch
import torch.nn as nn
import torch.nn.functional as F
from ..perception.simple_cnn import SplitNetCNN
from ..plan.rl_linear import AClinear, Qlinear
from vnenv.utils.net_utils import weights_init
import numpy as np


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
        q_flag=0
    ):
        super(SimpleMP, self).__init__()

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
            self.rct_shapes = {'lstm': (infer_sz, )}
            self.rct_dtypes = {'lstm': next(self.infer.parameters()).dtype}
        # plan
        if q_flag:
            self.plan_out = Qlinear(infer_sz, action_sz)
        else:
            self.plan_out = AClinear(infer_sz, action_sz)
        # self.apply(weights_init)

    def forward(self, vobs, tobs, rct):

        vobs_embed = F.relu(self.vobs_embed(vobs), True)
        tobs_embed = F.relu(self.tobs_embed(tobs), True)
        x = torch.cat((vobs_embed, tobs_embed), dim=1)
        x = self.drop(x)
        if self.mode == 0:
            x = self.infer(x)
            return self.plan_out(x)
        (x, cx) = self.infer(x, rct)
        out = self.plan_out(x)
        out.update(dict(lstm=(x, cx)))
        return out


class SplitLinear(torch.nn.Module):
    """简单模型1, splitnet + linear, 延时堆叠"""
    def __init__(
        self,
        action_sz,
        vobs_sz=(128, 128, 3),
        tobs_sz=300,
        obs_stack=1
    ):
        super(SplitLinear, self).__init__()
        self.obs_stack = obs_stack
        self.vobs_sz = vobs_sz
        # perception
        CNN = SplitNetCNN()
        self.conv_out_sz = CNN.out_fc_sz(vobs_sz[0], vobs_sz[1])
        self.vobs_conv = nn.Sequential(
            CNN,
            nn.Flatten(),
        )
        self.MP = SimpleMP(action_sz, self.conv_out_sz*obs_stack, tobs_sz)

    def forward(self, model_input):

        vobs = model_input['image|4'] \
            .view(-1, *self.vobs_sz).permute(0, 3, 1, 2)
        vobs = vobs / 255.

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed = torch.flatten(vobs_embed).view(
            -1, self.obs_stack*self.conv_out_sz
        )
        return self.MP(vobs_embed, model_input['glove'])


class SplitLstm(torch.nn.Module):
    """简单模型2,是简单模型1带LSTM的版本"""
    def __init__(
        self,
        action_sz,
        vobs_sz=(128, 128, 3),
        tobs_sz=300,
    ):
        super(SplitLstm, self).__init__()
        # perception
        self.vobs_sz = vobs_sz
        CNN = SplitNetCNN()
        self.conv_out_sz = CNN.out_fc_sz(vobs_sz[0], vobs_sz[1])
        self.vobs_conv = nn.Sequential(
            CNN,
            nn.Flatten(),
        )
        self.apply(weights_init)
        self.MP = SimpleMP(action_sz, self.conv_out_sz, tobs_sz, mode=1)
        self.rct_shapes = self.net.rct_shapes
        self.rct_dtypes = self.net.rct_dtypes

    def forward(self, model_input):

        vobs = model_input['image'].permute(0, 3, 1, 2) / 255.
        vobs_embed = self.vobs_conv(vobs)

        return self.MP(vobs_embed, model_input['glove'], model_input['hidden'])


class FcLstmModel(torch.nn.Module):
    """观察都是预处理好的特征向量的lstm模型"""
    def __init__(
        self,
        obs_shapes,
        act_sz,
        dropout_rate=0,
        q_flag=0
    ):
        super(FcLstmModel, self).__init__()
        self.net = SimpleMP(act_sz,
                            np.prod(obs_shapes['fc']),
                            np.prod(obs_shapes['glove']),
                            dropout_rate=dropout_rate,
                            mode=1, q_flag=q_flag)
        self.rct_shapes = self.net.rct_shapes
        self.rct_dtypes = self.net.rct_dtypes

    def forward(self, obs, rct):
        return self.net(
            obs['fc'],
            obs['glove'],
            rct['lstm']
        )


class FcLinearModel(torch.nn.Module):
    """观察都是预处理好的特征向量的linear模型"""
    def __init__(
        self,
        obs_shapes,
        act_sz
    ):
        super(FcLinearModel, self).__init__()
        self.vobs_sz = np.prod(obs_shapes['fc'])
        tobs_sz = np.prod(obs_shapes['glove'])

        self.net = SimpleMP(act_sz, self.vobs_sz, tobs_sz)

    def forward(self, obs, rct={}):

        vobs_embed = torch.flatten(obs['fc']) \
            .view(-1, self.vobs_sz)
        return self.net(vobs_embed, obs['glove'], rct)
