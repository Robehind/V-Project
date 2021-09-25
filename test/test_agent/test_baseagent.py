from vnenv.agents.base_agent import BaseAgent
import torch.nn as nn
import torch
import numpy as np
import pytest


class TESTmodel(nn.Module):
    def __init__(self) -> None:
        super(TESTmodel, self).__init__()
        self.f1 = nn.Linear(1, 1)
        self.step = -1
        self.outs = _acts
        self.rct_shapes = {'lstm': (6, )}
        self.rct_dtypes = {'lstm': torch.float32}
        self.rcts = torch.randn(5, 4, 6)

    def forward(self, obs, rct):
        self.step += 1
        out = torch.zeros((4, 5))
        for i, a in enumerate(self.outs[self.step]):
            out[i, a] = 999999
        return dict(policy=out,
                    rct={"lstm": self.rcts[self.step].clone()},
                    q_value=out,
                    action=torch.tensor(self.outs[self.step])
                    )


class TESTenv:
    def __init__(self) -> None:
        self.env_num = 4


@pytest.mark.parametrize(["select_func", "select_params"],
                         [
                             ['policy_select', []],
                             ['epsilon_select', [0.00001]],
                             ['continous', []]
                         ])
def test_base_agent(select_func, select_params):
    env = TESTenv()
    model = TESTmodel()
    agt = BaseAgent(model, env, None, select_func, select_params)
    assert np.allclose(agt.get_rct()['lstm'], 0)
    for i in range(5):
        done = np.array([0, 0, 0, 0])
        if i == 3:
            done = np.array([1, 0, 1, 0])
        idx = agt.action(dict(place=np.array([1, 2, 3])), done=done)
        if i == 3:
            assert np.allclose(agt.get_rct()['lstm'][[0, 2]], 0)
            assert np.allclose(agt.get_rct()['lstm'][[1, 3]],
                               model.rcts[i][[1, 3]])
        else:
            assert np.allclose(agt.get_rct()['lstm'], model.rcts[i])
        assert np.allclose(idx, np.array(_acts[i]))


_acts = [
    [3, 2, 0, 1],
    [3, 2, 3, 0],
    [3, 3, 3, 1],
    [1, 3, 1, 0],
    [4, 3, 4, 4]
]
