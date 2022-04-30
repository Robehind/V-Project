import numpy as np
from typing import Dict
from gym.vector import VectorEnv
from methods.utils.record_utils import MeanCalcer


class BaseRecorder:
    """写不同的recorder来记录特定的训练数据"""
    def __init__(
        self,
        Venv: VectorEnv,
        *args,
        **kwargs
    ) -> None:
        self.Venv = Venv
        self.env_num = Venv.num_envs
        self.mean_calc = MeanCalcer()

    def reset(self):
        self.mean_calc.pop()
        # log return and steps for each env
        self.env_return = np.zeros((self.env_num))
        self.env_steps = np.zeros((self.env_num))

    def record(self, r, done, info):
        # scalar records
        dones = done.sum()
        self.mean_calc.add(dict(epis=dones), count=False)
        self.env_return += r
        self.env_steps += 1
        for i in range(self.env_num):
            if done[i]:
                data = {
                    'ep_length': self.env_steps[i],
                    'return': self.env_return[i]}
                self.mean_calc.add(data)
                self.env_steps[i], self.env_return[i] = 0, 0

    def report(self) -> Dict:
        return self.mean_calc.report()

    def pop(self) -> Dict:
        """will reset all records and reset"""
        out = self.mean_calc.pop()
        if out == {}:
            return {'epis': 0}
        return out
