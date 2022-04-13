import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..select_funcs import policy_select, epsilon_select


class CartModel(nn.Module):
    def __init__(
        self,
        obs_spc,
        act_spc,
        q_flag=False,
        eps=0.08,
        *args, **kwargs
    ):
        super(CartModel, self).__init__()
        self.fc1 = nn.Linear(np.prod(obs_spc.shape), 256)
        self.fc_pi = nn.Linear(256, act_spc.n)
        self.fc_v = nn.Linear(256, 1)
        self.q_flag = q_flag
        self.eps = eps

    def forward(self, obs, rct={}):
        x = F.relu(self.fc1(obs['OBS']))
        if self.q_flag:
            out = dict(q_value=self.fc_pi(x))
            out['action'] = epsilon_select(self.eps, out).detach()
        else:
            out = dict(policy=self.fc_pi(x), value=self.fc_v(x))
            out['action'] = policy_select(out).detach()
        return out
