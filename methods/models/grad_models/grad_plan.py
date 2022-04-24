import torch
import torch.nn as nn
from ..plan import AClinear
from ..select_funcs import policy_select


class SplitDone(nn.Module):
    def __init__(
        self,
        ac_in_sz,
        act_n,
    ) -> None:
        super().__init__()
        self.ac = AClinear(ac_in_sz, act_n-1)

    def forward(self, x, done_tensor):
        out = self.ac(x)
        # 一定要保证done在最后一个
        out['policy'] = torch.cat([out['policy'], done_tensor], dim=1)
        out['action'] = policy_select(out).detach()
        return out
