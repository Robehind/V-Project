import torch.nn as nn
import torch.nn.functional as F


class CartModel(nn.Module):
    def __init__(
        self,
        obs_shapes,
        act_sz
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

    # def pi(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = self.fc_pi(x)
    #     # prob = F.softmax(x, dim=softmax_dim)
    #     return x

    # def v(self, x):
    #     x = F.relu(self.fc1(x))
    #     v = self.fc_v(x)
    #     return v
