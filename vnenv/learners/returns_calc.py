import numpy as np
# from numba import njit


# define some returns calculation functions for learner to use
# TODO bug fix @njit
def _basic_return(
    v_array: np.ndarray,
    last_v: np.ndarray,
    rew: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    nsteps: int
) -> np.ndarray:
    # input array should in (exp_length, exp_nums)
    R = last_v
    returns = np.zeros_like(rew)
    last_idx = returns.shape[0] - 1
    # TODO 所有项的gamma可以预先计算
    for i in range(last_idx, -1, -1):
        tau = i + nsteps
        if tau > last_idx:
            R = rew[i] + gamma * R * mask[i]
        else:
            R = v_array[tau]
            for j in range(nsteps):
                R = rew[tau-j-1] + gamma * R * mask[tau-j-1]
        returns[i] = R
    # output in (exp_length*exp_nums, 1)
    return returns.reshape(-1, 1)


def _GAE(
    v_array: np.ndarray,
    last_v: np.ndarray,
    rew: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    lbd: float,
    # nsteps: int
) -> np.ndarray:
    # TODO GAE暂时没有nsteps选项，但是理论上可以只累加未来的几个delta项
    if lbd == 1:
        return _basic_return(v_array, last_v, rew, mask, gamma, float("inf")) \
               - v_array.reshape(-1, 1)
    delta = _basic_return(v_array, last_v, rew, mask, gamma, 1) \
        - v_array.reshape(-1, 1)
    if lbd == 0:
        return delta
    exp_length = rew.shape[0]
    delta = delta.reshape(exp_length, -1)
    advs = np.zeros_like(rew)
    # TODO 所有项的衰减因子可以预先计算
    glbd = gamma * lbd
    advs[-1] = delta[-1]
    for i in range(exp_length - 2, -1, -1):
        advs[i] = glbd*advs[i+1]*mask[i] + delta[i]
    # output in (exp_length*exp_nums, 1)
    return advs.reshape(-1, 1)
