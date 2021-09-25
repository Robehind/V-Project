from typing import Generator, Union
import torch.nn.functional as F
import torch
import numpy as np


def continous(out: torch.Tensor) -> np.ndarray:
    act = out['action']
    # TODO noise?
    return act.numpy()


def epsilon_select(
    out: torch.Tensor,
    eps: Union[float, Generator]
) -> np.ndarray:
    if isinstance(eps, Generator):
        eps = next(eps)
    q = out['q_value'].cpu()
    a_idx = q.argmax(dim=1)
    r_idx = torch.randint_like(a_idx, high=q.shape[1])
    m = torch.rand(*a_idx.shape) <= eps
    a_idx[m] = r_idx[m]
    return a_idx.numpy()


def policy_select(out: torch.Tensor) -> np.ndarray:
    pi = out['policy'].cpu()
    prob = F.softmax(pi, dim=1)
    return prob.multinomial(1).numpy().squeeze(1)
