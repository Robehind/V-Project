from typing import Dict
from metrics import measure_epi
import re
import json


class EpisodeData:
    # 管理测试产生的数据，以一个episode为最小单位
    # 计算episode的相关metrics
    def __init__(
        self,
        json_path
    ) -> None:
        self.scenes = set()
        self.targets = set()
        self.models = set()
        self.steps = set()
        self.min_actses = set()
        self.episodes = []
        with open(json_path, 'r') as f:
            ori_epis = json.load(f)
        for epi in ori_epis:
            self.add_episode(epi)

    def add_episode(
        self,
        epi: Dict
    ) -> None:
        scene = epi['scene']
        target = epi['target']
        self.scenes.add(scene)
        self.targets.add(target)
        self.models.add(epi['model'])
        _epi = measure_epi(epi)
        self.episodes.append(_epi)

    def match(self, epi, attr, pattern):
        data = epi[attr]
        if re.match(pattern, str(data)):
            return True
        return False

    def get_episodes(self, re_dict: Dict):
        # re_dict为正则表达式字典，可以使用正则表达式筛选episode
        out = []
        for epi in self.episodes:
            flag = 1
            for k, v in re_dict.items():
                if not self.match(epi, k, v):
                    flag = 0
                    break
            if flag:
                out.append(epi.copy())
        return out
