from typing import Dict, Generator, Union
import torch.nn.functional as F
import torch
import numpy as np


def epsilon_select(
    eps: Union[float, Generator],
    out: Dict
) -> np.ndarray:
    q = out['q_value']
    a_idx = q.argmax(dim=1)
    r_idx = torch.randint_like(a_idx, high=q.shape[1])
    m = torch.rand(*a_idx.shape) <= eps
    a_idx[m] = r_idx[m]
    return a_idx


def policy_select(out: Dict) -> np.ndarray:
    pi = out['policy']
    prob = F.softmax(pi, dim=1)
    return prob.multinomial(1).squeeze(1)
