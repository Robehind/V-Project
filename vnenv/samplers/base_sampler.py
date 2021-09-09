from typing import Dict
import numpy as np
from .base_buffer import BaseBuffer
from vnenv.environments import VecEnv
from vnenv.agents import AbsAgent
from vnenv.curriculums import BaseCL
from vnenv.utils.record_utils import MeanCalcer


class BaseSampler:
    """用n个环境产生m*l大小的经验，即一个batch含有m条经验，每条经验长度都为l。
    """
    def __init__(
        self,
        Venv: VecEnv,
        Vagent: AbsAgent,
        CLscher: BaseCL,
        batch_size: int,
        exp_length: int,
        buffer_limit: int,
        buffer_update: int = None
    ) -> None:

        # if not setting buffer_update, then the buffer will be on-policy
        if buffer_update is None:
            buffer_update = buffer_limit
        assert buffer_update > 0
        assert batch_size % exp_length == 0

        self.Venv = Venv
        self.Vagent = Vagent
        self.cl = CLscher

        self.env_num = Venv.env_num
        self.exp_length = exp_length
        self.batch_size = batch_size
        sample_num = batch_size // exp_length

        if buffer_update // sample_num > 10:
            print(f"Warning: Need only {sample_num} exps to update model \
                but need to wait buffer update {buffer_update} exps. \
                Consider reduce buffer update number for better performance")

        self.rounds = buffer_update // self.env_num + \
            int(buffer_update % self.env_num != 0)

        # init curriculum
        CLscher.init_sche(sampler=self)

        # init buffer
        self.buffer = BaseBuffer(
            Venv.shapes,
            Venv.dtypes,
            exp_length,
            sample_num,
            self.env_num,
            buffer_limit
        )

        # init Mean Calcer
        self.mean_calc = MeanCalcer()

        # log rewards and steps for each env
        self.env_reward = np.zeros((self.env_num))
        self.env_steps = np.zeros((self.env_num))

        self.last_obs = self.Venv.reset()
        self.last_done = np.zeros((self.env_num))

    def sample(self) -> Dict:
        for _ in range(self.rounds):
            dones = self.run()
            # updating curriculum
            self.cl.next_sche(dones, self)
        return self.buffer.sample()

    def run(self) -> np.ndarray:
        # TODO agent能不在这里操作自己吗
        self.Vagent.clear_mems()
        for _ in range(self.exp_length):
            a_idx = self.Vagent.action(self.last_obs, self.last_done)
            obs_new, r, done, info = self.Venv.step(a_idx)
            # 记录o_t, a_t, r_t+1, m_t+1
            self.buffer.write_in(self.last_obs, a_idx, r, 1 - done)
            self.last_obs = obs_new
            self.last_done = done
            # scalar records
            dones = done.sum()
            self.mean_calc.add(dict(epis=dones), count=False)
            self.env_reward += r
            self.env_steps += 1
            for i in range(self.env_num):
                if done[i]:
                    self.mean_calc.add(dict(total_reward=self.env_reward[i]))
                    self.mean_calc.add(dict(total_steps=self.env_steps[i]))
                    self.mean_calc.add(dict(
                        success_rate=int(info[i]['success'])
                    ))
                    self.env_steps[i] = 0
                    self.env_reward[i] = 0
        # get one more obs to calc returns
        self.buffer.one_more_obs(self.last_obs)
        return dones

    def pop_records(self) -> Dict:
        """will reset all records and reset"""
        out = self.mean_calc.pop()
        if out == {}:
            return {'epis': 0}
        return out

    def close(self):
        self.Vagent.close()
        self.Venv.close()
