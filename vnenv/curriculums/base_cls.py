from typing import Dict
from .abs_cl import AbsCL
from random import shuffle
from vnenv.environments import VecEnv


# TODO 暂时是按照thor环境的接口来的
# 成功率达到一定地步就新增房间
class SceneCL(AbsCL):
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
        shuffle(self.all_scenes)
        self.crt_ed = init_num
        self.epi_cnt = epis_gate
        # init task
        settings = dict(chosen_scenes=self.all_scenes[:self.crt_ed])
        self.env.update_settings(settings)

    def next_task(self, update_steps: int, report: Dict):
        if 'success_rate' not in report:
            return False
        if self.crt_ed == len(self.all_scenes):
            return False
        if report['success_rate'] >= self.sr_gate:
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
