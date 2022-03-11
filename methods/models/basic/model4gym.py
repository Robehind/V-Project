import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CartModel(nn.Module):
    def __init__(
        self,
        obs_spc,
        act_spc,
        *args, **kwargs
    ):
        super(CartModel, self).__init__()
        self.fc1 = nn.Linear(np.prod(obs_spc.shape), 256)
        self.fc_pi = nn.Linear(256, act_spc.n)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, obs, rct={}):
        x = F.relu(self.fc1(obs['OBS']))
        return dict(
            policy=self.fc_pi(x),
            value=self.fc_v(x)
        )
