from gym.vector import VectorEnv
from .base_recorder import BaseRecorder


class TDNavRecorder(BaseRecorder):
    """用于目标驱动导航的recorder"""
    def __init__(
        self,
        Venv: VectorEnv,
        record_spl: bool
    ) -> None:
        super().__init__(Venv)
        self.Venv.call('add_extra_info', record_spl)

    def reset(self):
        super().reset()
        self.vis_cnt = []
        for info in self.Venv.call("info"):
            self.vis_cnt.append(info['visible'])

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
                self.env_steps[i], self.env_return[i] = 0, 0
                self.vis_cnt[i] = self.Venv.call("info")[i]['visible']
