import numpy as np
from numba import njit


# define some returns calculation functions for learner to use
@njit
def _basic_return(
    v_array: np.ndarray,
    rew: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    nsteps: int
):
    # input array should in (sample_steps, env_nums)
    # seperate last row as R
    v_array, R = v_array[:-1], v_array[-1]
    returns = np.zeros_like(rew)
    last_idx = returns.shape[0] - 1
    # TODO 所有项的gamma可以预先计算
    for i in range(last_idx, -1, -1):
        tau = i+nsteps
        if tau > last_idx:
            R = rew[i] + gamma * R * mask[i]
        else:
            R = v_array[tau]
            for i in range(nsteps):
                R = rew[tau-i-1] + gamma * R * mask[tau-i-1]
        returns[i] = R
    return returns.reshape(-1, 1)
