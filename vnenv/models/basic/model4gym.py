import torch.nn as nn
import torch.nn.functional as F


class CartModel(nn.Module):
    def __init__(
        self, *args, **kwargs
    ):
        super(CartModel, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, obs, rct={}):
        x = F.relu(self.fc1(obs['obs']))
        return dict(
            policy=self.fc_pi(x),
            value=self.fc_v(x)
        )
