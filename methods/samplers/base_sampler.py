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
        agent: AbsAgent,
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
        # init Mean Calcer
        self.mean_calc = MeanCalcer()
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
        self.mean_calc.pop()
        # log return and steps for each env
        self.env_return = np.zeros((self.env_num))
        self.env_steps = np.zeros((self.env_num))

        self.last_obs = self.Venv.reset()
        self.vis_cnt = []
        for info in self.Venv.call("info"):
            self.vis_cnt.append(info['visible'])
        self.last_done = np.ones((self.env_num))

    def sample(self) -> Dict:
        for _ in range(self.rounds):
            self.run(self.exp_length+1)
        return self.buffer.sample()

    def record(self, r, done, info):
        # scalar records
        dones = done.sum()
        self.mean_calc.add(dict(epis=dones), count=False)
        self.env_return += r
        self.env_steps += 1
        for i in range(self.env_num):
            self.vis_cnt[i] += info[i]['visible']
            if done[i]:
                self.vis_cnt[i] -= info[i]['visible']
                data = {
                    'ep_length': self.env_steps[i],
                    'SR': int(info[i]['success']),
                    'return': self.env_return[i],
                    'vis_cnt': self.vis_cnt[i]}
                # 只要环境反馈了最短路信息，那么就算一下SPL
                if 'min_acts' in info[i]:
                    spl = 0
                    if info[i]['success']:
                        assert info[i]['min_acts'] <= self.env_steps[i],\
                            f"{info[i]['min_acts']}>{self.env_steps[i]}"
                        # TODO spl计算问题。0？done？
                        spl = info[i]['min_acts']/self.env_steps[i]
                    data['SPL'] = spl
                self.mean_calc.add(data)
                # if 'success' in info[i]:
                #     self.mean_calc.add(dict(SR=int(info[i]['success'])))
                self.env_steps[i], self.env_return[i] = 0, 0
                self.vis_cnt[i] = self.Venv.call("info")[i]['visible']

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
            self.record(r, done, info)

    def report(self) -> Dict:
        return self.mean_calc.report()

    def pop_records(self) -> Dict:
        """will reset all records and reset"""
        out = self.mean_calc.pop()
        if out == {}:
            return {'epis': 0}
        return out

    def close(self):
        self.agent.close()
        self.Venv.close()
