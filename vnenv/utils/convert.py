from typing import Dict, Optional, Union
import torch
import numpy as np


def dtype2numpy(
    dtype: Union[np.dtype, type, torch.dtype]
) -> np.dtype:
    if isinstance(dtype, type):
        return np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        return dtype
    return torch.zeros((1,), dtype=dtype).numpy().dtype


def dtype2tensor(
    dtype: Union[np.dtype, type, torch.dtype]
) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.from_numpy(np.zeros((1,), dtype=dtype)).dtype


def dict2tensor(
    x: Dict[str, np.ndarray],
    dev: torch.device = "cpu",
    dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    o = {}
    for k in x:
        o[k] = toTensor(
            x[k], dev, dtype=dtype
        )
    return o


def dict2numpy(
    x: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    o = {}
    for k in x:
        o[k] = toNumpy(x[k])
    return o


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
