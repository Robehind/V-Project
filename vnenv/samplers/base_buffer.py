from typing import Dict, Union
import numpy as np
from vnenv.utils.convert import dtype2numpy


class Buff:
    """"""
    def __init__(
        self,
        dshapes: Dict[str, tuple],
        dtypes: Dict[str, np.dtype],
        bshapes: Dict[str, tuple]
    ) -> None:
        self.dshapes = dshapes
        self.dtypes = dtypes
        self.bshapes = bshapes
        self.buff = {
            k: np.zeros((*bshapes[k], *v), dtype=dtype2numpy(dtypes[k]))
            for k, v in dshapes.items()
        }


class BaseBuffer(Buff):
    """"""
    def __init__(
        self,
        obs_shapes: Dict[str, tuple],
        obs_dtypes: Dict[str, np.dtype],
        rct_shapes: Dict[str, tuple],
        rct_dtypes: Dict[str, np.dtype],
        exp_length: int,
        sample_num: int,
        env_num: int,
        max_exp_num: int
    ) -> None:
        assert env_num <= max_exp_num
        # buffer for r，a，mask and observation and rcts
        shapes = {
            "r": (),  # TODO if (1, ) can't broadcast (x) to (x, 1)
            "m": (),
            "a": (),
        }
        shapes.update(obs_shapes)
        shapes.update(rct_shapes)
        dtypes = {
            "r": np.float32,
            "m": np.int8,
            "a": np.int64,
        }
        dtypes.update(obs_dtypes)
        dtypes.update(rct_dtypes)
        # buff shapes
        bshapes = {}
        for k in shapes:
            bshapes[k] = (exp_length, max_exp_num)
        # exp_length for obs plus 1 for extral value calc
        for k in obs_shapes:
            bshapes[k] = (exp_length + 1, max_exp_num)

        super().__init__(shapes, dtypes, bshapes)

        self.exp_length = exp_length
        self.obs_shapes = obs_shapes
        self.rct_shapes = rct_shapes
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
        obs: Dict[str, np.ndarray]
    ):
        ed = self.exp_p + self.env_num
        # if max_exp_num can't be divided exactly by env_num
        # then the write_in process will sometimes be splited
        if ed > self.max_exp_num:
            spt1 = self.max_exp_num - self.exp_p
            for k, v in obs.items():
                self.buff[k][-1][self.exp_p:] = v[:spt1]
            for k, v in obs.items():
                self.buff[k][-1][:self.env_num-spt1] = v[spt1:]
        else:
            for k, v in obs.items():
                self.buff[k][-1][self.exp_p:ed] = v
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
        obs: Dict[str, np.ndarray],
        rct: Dict[str, np.ndarray],
        a: np.ndarray,
        r: np.ndarray,
        m: np.ndarray
    ) -> None:
        # no matter what, just write in
        for k, v in obs.items():
            self.buff[k][self.step_p][st:ed] = v
        for k, v in rct.items():
            self.buff[k][self.step_p][st:ed] = v
        self.buff['r'][self.step_p][st:ed] = r
        self.buff['m'][self.step_p][st:ed] = m
        self.buff['a'][self.step_p][st:ed] = a

    def write_in(
        self,
        obs: Dict[str, np.ndarray],
        rct: Dict[str, np.ndarray],
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
                {k: v[:spt1] for k, v in obs.items()},
                {k: v[:spt1] for k, v in rct.items()},
                a[:spt1],
                r[:spt1],
                m[:spt1]
                )
            self._write_in(
                0,
                self.env_num - spt1,
                {k: v[spt1:] for k, v in obs.items()},
                {k: v[spt1:] for k, v in rct.items()},
                a[spt1:],
                r[spt1:],
                m[spt1:]
                )
        else:
            self._write_in(self.exp_p, ed, obs, rct, a, r, m)
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
            'obs': {
                k: self.buff[k][:, idx]
                for k in self.obs_shapes
            },
            'rct': {
                k: self.buff[k][:, idx]
                for k in self.rct_shapes
            },
            'a': self.buff['a'][:, idx],
            # don't flatten r and m for faster numpy mat computation
            'r': self.buff['r'][:, idx],
            'm': self.buff['m'][:, idx]
        }
