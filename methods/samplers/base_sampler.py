import numpy as np
from typing import Dict
from .base_buffer import BaseBuffer
from gym.vector import VectorEnv
from methods.agents import AbsAgent
from methods.utils.record_utils import MeanCalcer


class BaseSampler:
    """用n个环境产生m*l大小的经验，即一个batch含有m条经验，每条经验长度都为l。
    """
    def __init__(
        self,
        Venv: VectorEnv,
        Vagent: AbsAgent,
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

        self.env_num = Venv.num_envs
        self.exp_length = exp_length
        self.batch_size = batch_size
        sample_num = batch_size // exp_length

        if buffer_update // sample_num > 10:
            print(f"Warning: Need only {sample_num} exps to update model \
                but need to wait buffer update {buffer_update} exps. \
                Consider reduce buffer update number for better performance")

        self.rounds = buffer_update // self.env_num + \
            int(buffer_update % self.env_num != 0)

        # init buffer
        self.buffer = BaseBuffer(
            Venv.single_observation_space,
            Vagent.rct_shapes,
            Vagent.rct_dtypes,
            exp_length + 1,  # sample_length = exp_length + 1
            sample_num,
            self.env_num,
            buffer_limit
        )

        # init Mean Calcer
        self.mean_calc = MeanCalcer()
        self.reset()

    def reset(self):
        self.buffer.clear()
        self.mean_calc.pop()
        # log rewards and steps for each env
        self.env_reward = np.zeros((self.env_num))
        self.env_steps = np.zeros((self.env_num))

        self.last_obs = self.Venv.reset()
        self.last_done = np.ones((self.env_num))

    def sample(self) -> Dict:
        for _ in range(self.rounds):
            self.run()
        return self.buffer.sample()

    def run(self) -> np.ndarray:
        # sample exp_length + 1 exps for learner's need
        for _ in range(self.exp_length+1):
            a_idx, last_rct = self.Vagent.action(self.last_obs, self.last_done)
            obs_new, r, done, info = self.Venv.step(a_idx)
            # record obs_t, rct_t, a_t, r_t+1, m_t+1
            self.buffer.write_in(
                self.last_obs, last_rct,
                a_idx, r, 1 - done
            )
            self.last_obs = obs_new
            self.last_done = done
            # scalar records
            dones = done.sum()
            self.mean_calc.add(dict(epis=dones), count=False)
            self.env_reward += r
            self.env_steps += 1
            for i in range(self.env_num):
                if done[i]:
                    self.mean_calc.add({'return': self.env_reward[i]})
                    self.mean_calc.add(dict(ep_length=self.env_steps[i]))
                    if 'success' in info[i]:
                        self.mean_calc.add(dict(SR=int(info[i]['success'])))
                    self.env_steps[i] = 0
                    self.env_reward[i] = 0
        return dones

    def report(self) -> Dict:
        return self.mean_calc.report()

    def pop_records(self) -> Dict:
        """will reset all records and reset"""
        out = self.mean_calc.pop()
        if out == {}:
            return {'epis': 0}
        return out

    def close(self):
        self.Vagent.close()
        self.Venv.close()