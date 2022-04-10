import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.net_utils import norm_col_init


# 输出是用dict封装的
class Qlinear(torch.nn.Module):
    """输出Q的输出层, 挂一个激活函数的"""
    def __init__(
        self,
        input_sz,
        action_sz
    ):
        super().__init__()
        self.q_linear = nn.Linear(input_sz, action_sz)
        self.q_linear.weight.data = norm_col_init(
            self.q_linear.weight.data, 1.0)
        self.q_linear.bias.data.fill_(0)

    def forward(self, x, rct=None):
        return dict(q_value=self.q_linear(F.relu(x)))


class AClinear(torch.nn.Module):
    """Actor-critic的输出层, 挂一个激活函数的"""
    def __init__(
        self,
        input_sz,
        action_sz
    ):
        super(AClinear, self).__init__()
        self.actor_linear = nn.Linear(input_sz, action_sz)
        self.critic_linear = nn.Linear(input_sz, 1)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x, rct=None):
        x = F.relu(x)
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x))
