from typing import Dict
from .abs_cl import AbsCL
from vnenv.environments import VecEnv
import random


class SceneCL(AbsCL):
    # TODO 暂时是按照thor环境的接口来的
    # 成功率达到一定地步就新增房间
    def __init__(
        self,
        env: VecEnv,
        init_num: int = 1,
        incre_num: int = 1,
        epis_gate: int = 200,
        sr_gate: float = 0.8
    ) -> None:
        self.env = env
        self.epis_gate = epis_gate
        self.sr_gate = sr_gate
        self.incre_num = incre_num

        self.all_scenes = env.chosen_scenes.copy()
        random.shuffle(self.all_scenes)
        self.crt_ed = init_num
        self.epi_cnt = epis_gate
        # init task
        settings = dict(chosen_scenes=self.all_scenes[:self.crt_ed])
        self.env.update_settings(settings)

    def next_task(self, update_steps: int, report: Dict):
        if 'SR' not in report:
            return False
        if self.crt_ed == len(self.all_scenes):
            return False
        if report['SR'] >= self.sr_gate:
            self.epi_cnt -= report['epis']
            if self.epi_cnt <= 0:
                self.epi_cnt = self.epis_gate
                self.crt_ed += self.incre_num
                self.crt_ed = min(self.crt_ed, len(self.all_scenes))
                settings = dict(chosen_scenes=self.all_scenes[:self.crt_ed])
                self.env.update_settings(settings)
                return True
            return False
        else:
            self.epi_cnt = self.epis_gate
            return False


class UniSceneCL(AbsCL):
    # 为不同的进程设定不同房间集合
    def __init__(
        self,
        env: VecEnv,
        shuffle: bool = True
    ) -> None:
        n = env.env_num
        scenes = env.chosen_scenes.copy()
        assert n <= len(scenes)
        if shuffle:
            random.shuffle(scenes)
        self.settings = []
        step = len(scenes)//n
        mod = len(scenes) % n
        for i in range(0, n*step, step):
            self.settings.append(scenes[i:i + step])
        for i in range(0, mod):
            self.settings[i].append(scenes[-(i+1)])
        # init task
        for i in range(n):
            settings = dict(chosen_scenes=self.settings[i])
            env.update_settings_proc(i, settings)

    def next_task(self, update_steps: int, report: Dict):
        return False
