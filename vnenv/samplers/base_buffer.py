from typing import Dict, Union
import numpy as np


class BaseBuffer:
    """"""
    def __init__(
        self,
        obs_Dshape: Dict[str, tuple],
        obs_Dtype: Dict[str, np.dtype],
        exp_length: int,
        sample_num: int,
        env_num: int,
        max_exp_num: int
    ) -> None:
        assert env_num <= max_exp_num
        # obs维度要多1，因为有的算法需要一个额外的obs来计算return
        # buffer for r，a，mask and observation
        self.reward = np.zeros((exp_length, max_exp_num), dtype=np.float32)
        # TODO a和m改成更小的数据类型是不是会更快？
        self.a_idx = np.zeros((exp_length, max_exp_num), dtype=np.int64)
        self.mask = np.zeros((exp_length, max_exp_num), dtype=np.int8)
        self.obs = {
            k: np.zeros((exp_length + 1, max_exp_num, *v), dtype=obs_Dtype[k])
            for k, v in obs_Dshape.items()
        }

        self.exp_length = exp_length
        self.obs_Dshape = obs_Dshape
        self.max_exp_num = max_exp_num
        self.env_num = env_num
        self.sample_num = sample_num

        # pointer to control the write in position
        self.step_p = 0
        self.exp_p = 0
        # once full flag
        self.once_full = False

    def one_more_obs(
        self,
        o: Dict[str, np.ndarray]
    ):
        ed = self.exp_p + self.env_num
        # if max_exp_num can't be divided exactly by env_num
        # then the write_in process will sometimes be splited
        if ed > self.max_exp_num:
            spt1 = self.max_exp_num - self.exp_p
            for k, v in o.items():
                self.obs[k][-1][self.exp_p:] = v[:spt1]
            for k, v in o.items():
                self.obs[k][-1][:self.env_num-spt1] = v[spt1:]
        else:
            for k, v in o.items():
                self.obs[k][-1][self.exp_p:ed] = v
        if self.step_p == self.exp_length:
            self.step_p = 0
            self.exp_p += self.env_num
            if self.exp_p >= self.max_exp_num:
                self.once_full = True
                self.exp_p %= self.max_exp_num

    def _write_in(
        self,
        st: int,
        ed: int,
        o: Dict[str, np.ndarray],
        a: np.ndarray,
        r: np.ndarray,
        m: np.ndarray
    ) -> None:
        # no matter what, just write in
        for k, v in o.items():
            self.obs[k][self.step_p][st:ed] = v
        self.reward[self.step_p][st:ed] = r
        self.mask[self.step_p][st:ed] = m
        self.a_idx[self.step_p][st:ed] = a

    def write_in(
        self,
        o: Dict[str, np.ndarray],
        a: np.ndarray,
        r: np.ndarray,
        m: np.ndarray
    ) -> None:
        ed = self.exp_p + self.env_num
        # if max_exp_num can't be divided exactly by env_num
        # then the write_in process will sometimes be splited
        if ed > self.max_exp_num:
            spt1 = self.max_exp_num - self.exp_p
            self._write_in(
                self.exp_p,
                self.max_exp_num,
                {k: v[:spt1] for k, v in o.items()},
                a[:spt1],
                r[:spt1],
                m[:spt1]
                )
            self._write_in(
                0,
                self.env_num - spt1,
                {k: v[spt1:] for k, v in o.items()},
                a[spt1:],
                r[spt1:],
                m[spt1:]
                )
        else:
            self._write_in(self.exp_p, ed, o, a, r, m)
        self.step_p += 1

    def sample(
        self
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        if not self.once_full:
            # if buffer not fully filled, then sample from filled cols
            assert self.exp_p >= self.sample_num,\
                f"Don't have enough exps in buffer \
                 ({self.exp_p+1} vs {self.sample_num})"
            tmp = np.arange(self.exp_p)
        else:
            tmp = np.arange(self.max_exp_num)
        # TODO maybe slow
        np.random.shuffle(tmp)
        idx = tmp[:self.sample_num]
        return {
            'o': {
                k: self.obs[k][:, idx].reshape(-1, *v)
                for k, v in self.obs_Dshape.items()
            },
            'a': self.a_idx[:, idx].reshape(-1, 1),
            # don't flatten r and m for faster numpy mat computation
            'r': self.reward[:, idx],
            'm': self.mask[:, idx]
        }
