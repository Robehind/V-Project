import torch.nn as nn
from gym.spaces import Discrete
import torch
import numpy as np


class DoneNet(nn.Module):
    def __init__(
        self,
        feat_sz,
        mid_sz,
        dprate
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_sz, mid_sz),
            nn.ReLU(),
            nn.Dropout(p=dprate),
            nn.Linear(mid_sz, 1),
            nn.Sigmoid())

    def forward(self, feat):
        return self.net(feat)


def done_wrapper(name, clss):
    class OurModel(clss):
        """TgtAttActMat + DoneNet"""
        def __init__(
            self,
            obs_spc,
            act_spc,
            learnable_x,
            init,
            done_net_path,
            done_thres,
        ):
            _act_spc = Discrete(act_spc.n-1)
            super().__init__(obs_spc, _act_spc, learnable_x, init)
            self.done_thres = done_thres
            self.done_idx = act_spc.n - 1
            fc_sz = np.prod(obs_spc['fc'].shape)
            wd_sz = np.prod(obs_spc['wd'].shape)
            self.done_net = DoneNet(fc_sz+wd_sz, 512, 0)
            self.done_net.requires_grad_(False)
            self.done_net.load_state_dict(torch.load(done_net_path))
            self.done_net.eval()
            self.big_num = 9999999

        def forward(self, obs, rct):
            out = super().forward(obs, rct)
            action = out['action'].unsqueeze(1)
            # done plan
            with torch.no_grad():
                done = self.done_net(torch.cat([obs['fc'], obs['wd']], dim=1))
            idx = done >= self.done_thres
            action[idx] = self.done_idx
            # 加一行policy凑够动作，done是不带梯度的
            inf_pre = -torch.ones_like(done)
            inf_pre[idx] = -inf_pre[idx]
            out['policy'] = torch.cat(
                [out['policy'], self.big_num*inf_pre], dim=1)
            out['action'] = action.squeeze()
            return out
    return type(name, (OurModel, ), {})
