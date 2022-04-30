import numpy as np
from typing import Dict
from .base_buffer import BaseBuffer
from .base_recorder import BaseRecorder
from gym.vector import VectorEnv
from methods.agents import AbsAgent


class BaseSampler:
    """用n个环境产生m*l大小的经验，即一个batch含有m条经验，每条经验长度都为l。
    """
    def __init__(
        self,
        Venv: VectorEnv,
        agent: AbsAgent,
        recorder: BaseRecorder,
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
        self.buffer_limit = buffer_limit

        self.Venv = Venv
        self.agent = agent
        self.recorder = recorder

        self.env_num = Venv.num_envs
        self.exp_length = exp_length
        self.batch_size = batch_size
        self.sample_num = batch_size // exp_length

        if buffer_update // self.sample_num > 10:
            print(f"Warning: Need only {self.sample_num} exps to update model \
                but need to wait buffer update {buffer_update} exps. \
                Consider reduce buffer update number for better performance")

        self.rounds = buffer_update // self.env_num + \
            int(buffer_update % self.env_num != 0)

        self.init_buffer()
        self.reset()

    def init_buffer(self):
        # init buffer
        self.buffer = BaseBuffer(
            self.Venv.single_observation_space,
            self.agent.rct_shapes,
            self.agent.rct_dtypes,
            self.exp_length + 1,  # sample_length = exp_length + 1
            self.sample_num,
            self.env_num,
            self.buffer_limit)

    def reset(self):
        self.buffer.clear()
        self.last_obs = self.Venv.reset()
        self.last_done = np.ones((self.env_num))
        self.recorder.reset()

    def sample(self) -> Dict:
        for _ in range(self.rounds):
            self.run(self.exp_length+1)
        return self.buffer.sample()

    def run(self, length):
        # sample exp_length + 1 exps for learner's need
        for _ in range(length):
            a_idx, last_rct = self.agent.action(self.last_obs, self.last_done)
            obs_new, r, done, info = self.Venv.step(a_idx)
            # record obs_t, rct_t, a_t, r_t+1, m_t+1
            self.buffer.write_in(
                self.last_obs, last_rct,
                a_idx, r, 1 - done)
            self.last_obs = obs_new
            self.last_done = done
            # scalar records
            self.recorder.record(r, done, info)

    def report(self) -> Dict:
        return self.recorder.report()

    def pop_records(self) -> Dict:
        """will reset all records and reset"""
        return self.recorder.pop()

    def close(self):
        self.agent.close()
        self.Venv.close()
