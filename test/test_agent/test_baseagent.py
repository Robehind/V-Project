import glob
import os
from vnenv.agents.base_agent import BaseAgent
from vnenv.utils.init_func import set_seed
import torch.nn as nn
import torch
import numpy as np


class TESTmodel(nn.Module):
    def __init__(self) -> None:
        super(TESTmodel, self).__init__()
        self.f1 = nn.Linear(1, 1)
        self.step = -1
        self.outs = _acts

    def forward(self, obs):
        self.step += 1
        out = torch.zeros((4, 5))
        for i, a in enumerate(self.outs[self.step]):
            out[i, a] = 999999
        return dict(policy=out)


class TESTenv:
    def __init__(self) -> None:
        self.env_num = 4


def test_base_agent():
    set_seed(1114)
    env = TESTenv()
    model = TESTmodel()
    agt = BaseAgent(model, env)
    for i in range(5):
        idx = agt.action(dict(place=np.array([1, 2, 3])))
        assert np.allclose(idx, np.array(_acts[i]))
    agt.save_model('./', 100)
    ss = glob.glob('./*.dat')
    assert len(ss) == 1
    os.remove(ss[0])


_acts = [
    [3, 2, 0, 1],
    [3, 2, 3, 0],
    [3, 3, 3, 1],
    [1, 3, 1, 0],
    [4, 3, 4, 4]
]
if __name__ == '__main__':
    test_base_agent()
