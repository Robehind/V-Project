import torch.nn as nn
import torch.nn.functional as F
from .plan.rl_linear import AClinear
import numpy as np
import torch


class DemoModel(nn.Module):
    """观察都是预处理好的特征向量的linear模型"""
    def __init__(
        self,
        obs_shapes,
        act_sz
    ):
        super(DemoModel, self).__init__()
        self.obs_sz = np.prod(obs_shapes['map'])

        self.embed = nn.Linear(self.obs_sz, 256)
        self.net = nn.Linear(256, 256)
        self.plan = AClinear(act_sz, 256)

    def forward(self, model_input):
        mapp = torch.flatten(model_input['map']) \
            .view(-1, self.obs_sz)
        x = F.relu(self.embed(mapp))
        x = F.relu(self.net(x))
        return self.plan(x)
