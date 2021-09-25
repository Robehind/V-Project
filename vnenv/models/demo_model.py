import torch.nn as nn
import torch.nn.functional as F
from .plan.rl_linear import AClinear
import numpy as np
import torch


# rule1: don't activate the input, do acivate the output
# rule2: forward has 2 params: obs and rct, rct for recurrent inputs,
#        keys shouldn't repeat
#        the output has a key "rct" to contain the rct data
# rule3: if has recurrent inputs, define rct_shapes attr and rct_dtypes
class DemoModel(nn.Module):
    """A model for demo and a template for model definition"""
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

    def forward(self, obs, rct=None):
        mapp = torch.flatten(obs['map']) \
            .view(-1, self.obs_sz)
        x = F.relu(self.embed(mapp))
        x = F.relu(self.net(x))
        return self.plan(x)
