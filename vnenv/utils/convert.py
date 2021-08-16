from typing import Optional
import torch
import numpy as np


def toNumpy(x: torch.Tensor):
    """convert torch.Tensor to numpy"""
    return x.detach().cpu().numpy()


def toTensor(
    x: np.ndarray,
    dev: torch.device = "cpu",
    dtype: Optional[torch.dtype] = None
):
    """convert numpy to torch.Tensor"""
    x = torch.from_numpy(x).to(dev)
    if dtype is None:
        return x
    return x.to(dtype)
