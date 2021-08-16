from typing import Dict, Union
import numpy as np


class BaseBuffer:

    def __init__(
        self,
        obs_Dshape: Dict[str, tuple],
        obs_Dtype: Dict[str, np.dtype],
        env_num: int,
        sample_steps: int
    ) -> None:
        # buffer for r，a，mask and observation
        self.reward = np.zeros((sample_steps, env_num), dtype=np.float32)
        # TODO a和m改成更小的数据类型是不是会更快？
        self.a_idx = np.zeros((sample_steps, env_num), dtype=np.int64)
        self.mask = np.zeros((sample_steps, env_num), dtype=np.int8)
        # obs的维度要多1，因为需要一个额外的obs来计算return
        self.obs = {
            k: np.zeros((sample_steps + 1, env_num, *v), dtype=obs_Dtype[k])
            for k, v in obs_Dshape.items()
        }
        self.p = 0
        self.sample_steps = sample_steps
        self.obs_Dshape = obs_Dshape

    def one_more_obs(
        self,
        o: Dict[str, np.ndarray]
    ):
        for k, v in o.items():
            self.obs[k][-1] = v

    def write_in(
        self,
        o: Dict[str, np.ndarray],
        a: np.ndarray,
        r: np.ndarray,
        m: np.ndarray
    ) -> None:
        for k, v in o.items():
            self.obs[k][self.p] = v
        self.reward[self.p] = r
        self.mask[self.p] = m
        self.a_idx[self.p] = a
        self.p += 1

    def batched_out(
        self
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        # 算return的时候好像不用把r，m摊平更快
        self.p %= self.sample_steps
        if self.p != 0:
            print('warning: read the buffer when not fully filled')
        return {
            'o': {
                k: self.obs[k].reshape(-1, *v)
                for k, v in self.obs_Dshape.items()
            },
            'a': self.a_idx.reshape(-1, 1),
            'r': self.reward,
            'm': self.mask
        }
