import numpy as np


def np_gather2d(
    x: np.ndarray,
    idx: np.ndarray
) -> np.ndarray:
    x_ind = np.tile(np.arange(len(x)), (idx.shape[1], 1)).transpose()
    return x[x_ind, idx]
