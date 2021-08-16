from typing import Dict
import numpy as np
from .base_buffer import BaseBuffer
from vnenv.environments import VecEnv
from vnenv.agents import AbsAgent
from vnenv.curriculums import BaseCL
from vnenv.utils.record_utils import MeanCalcer


class BaseSampler:
    """管理向量化的环境和向量化的智能体进行交互，产生大于等于batch_size时间步的经验
       记录数据，数据类型都为np.ndarray
       管理课程学习类
    """
    def __init__(
        self,
        Venv: VecEnv,
        Vagent: AbsAgent,
        CLscher: BaseCL,
        batch_size: int
    ) -> None:
        self.batch_size = batch_size
        self.Venv = Venv
        self.Vagent = Vagent
        self.cl = CLscher

        # 在batch size一定的情况下，环境的个数将影响单个环境的采样步数
        self.env_num = self.Venv.env_num
        assert batch_size % self.env_num == 0
        self.sample_steps = batch_size // self.env_num

        # init curriculum
        CLscher.init_sche(self)

        # init buffer
        self.buffer = BaseBuffer(
            Venv.shapes,
            Venv.dtypes,
            self.env_num,
            self.sample_steps
        )

        # init Mean Calcer
        self.mean_calc = MeanCalcer()

        # log rewards and steps for each env
        self.env_reward = np.zeros((self.env_num))
        self.env_steps = np.zeros((self.env_num))

        self.last_obs = self.Venv.reset()
        self.last_done = np.zeros((self.env_num))

    def run(self) -> Dict:
        # TODO agent能不在这里操作自己吗
        self.Vagent.clear_mems()
        for _ in range(self.sample_steps):
            a_idx = self.Vagent.action(self.last_obs, self.last_done)
            obs_new, r, done, info = self.Venv.step(a_idx)
            # 记录o_t, a_t, r_t+1, m_t+1
            self.buffer.write_in(self.last_obs, a_idx, r, 1 - done, )
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
                        success_rate=int(info[i]['event'] == 'success')
                    ))
                    self.env_steps[i] = 0
                    self.env_reward[i] = 0
        # get one more obs to calc returns
        self.buffer.one_more_obs(self.last_obs)
        # updating curriculum
        self.cl.next_sche(dones, self)
        return self.buffer.batched_out()

    def pop_records(self) -> Dict:
        """will reset all records and reset"""
        out = self.mean_calc.pop()
        if out == {}:
            return {'epis': 0}
        return out

    def close(self):
        self.Vagent.close()
        self.Venv.close()
