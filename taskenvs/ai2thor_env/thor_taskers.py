from taskenvs.tasker import Tasker
from gym.vector import VectorEnv
from typing import Optional, Dict
from .utils import get_scene_names
import random
from copy import deepcopy
from collections import deque


class ThorAveSceneTasker(Tasker):
    """Scene task space equally divided into env nums"""
    def __init__(
        self,
        envs: VectorEnv,
        tasks: Optional[Dict] = None
    ):
        if tasks is None:
            return
        scenes = get_scene_names(tasks['scenes'])
        n = envs.num_envs
        random.shuffle(scenes)
        if n > len(scenes):
            out = [scenes for _ in range(n)]
        else:
            out = []
            step = len(scenes)//n
            mod = len(scenes) % n
            for i in range(0, n*step, step):
                out.append(scenes[i:i + step])
            for i in range(0, mod):
                out[i].append(scenes[-(i+1)])
        task_list = [deepcopy(tasks) for _ in range(n)]
        for i, t in enumerate(task_list):
            t['scenes'] = out[i]
        envs.set_attr('tasks', task_list)


class SceneScaleTasker(Tasker):
    """the scale of scene grows"""
    def __init__(
        self,
        envs: VectorEnv,
        tasks: Optional[Dict],
        window_sz: int,
        inc_num: int,
        sr_gate: float,
        epis_gate: int
    ):
        if tasks is None:
            return
        self.envs = envs
        self.tasks = tasks.copy()
        self.epis_gate = epis_gate
        self.sr_gate = sr_gate
        self.sr_que = deque(maxlen=epis_gate)
        self.inc_num = inc_num
        self.window_sz = window_sz
        scenes = get_scene_names(tasks['scenes'])
        self.st = 0
        # 按规模排序
        self.s_scenes = sorted(scenes, key=lambda x: scene_scales[x])
        self.s_nums = len(self.s_scenes)
        _task = tasks.copy()
        _task['scenes'] = self.s_scenes[:window_sz]
        envs.set_attr('tasks', _task)

    def next_tasks(self, update_steps, report):
        for _ in range(int(report.pop('epis'))):
            self.sr_que.append(report['SR'])
        if len(self.sr_que) < self.epis_gate or \
           self.st+self.window_sz == self.s_nums or \
           sum(self.sr_que)/self.epis_gate < self.sr_gate:
            return False
        _task = self.tasks.copy()
        self.st += self.inc_num
        if self.st+self.window_sz > self.s_nums:
            self.st -= (self.st+self.window_sz - self.s_nums)
        _task['scenes'] = self.s_scenes[self.st:self.st+self.window_sz]
        self.envs.set_attr('tasks', _task)
        return True


scene_scales = {
    "FloorPlan1": 2592,
    "FloorPlan2": 2544,
    "FloorPlan3": 1840,
    "FloorPlan4": 1376,
    "FloorPlan5": 2032,
    "FloorPlan6": 2336,
    "FloorPlan7": 5184,
    "FloorPlan8": 3344,
    "FloorPlan9": 1536,
    "FloorPlan10": 4304,
    "FloorPlan11": 1264,
    "FloorPlan12": 1744,
    "FloorPlan13": 3616,
    "FloorPlan14": 2448,
    "FloorPlan15": 1856,
    "FloorPlan16": 3568,
    "FloorPlan17": 1424,
    "FloorPlan18": 4176,
    "FloorPlan19": 1360,
    "FloorPlan20": 1760,
    "FloorPlan21": 1840,
    "FloorPlan22": 2688,
    "FloorPlan23": 2144,
    "FloorPlan24": 1360,
    "FloorPlan25": 576,
    "FloorPlan26": 1504,
    "FloorPlan27": 880,
    "FloorPlan28": 1856,
    "FloorPlan29": 1568,
    "FloorPlan30": 1376,
    "FloorPlan201": 3744,
    "FloorPlan202": 2880,
    "FloorPlan203": 9488,
    "FloorPlan204": 3968,
    "FloorPlan205": 4752,
    "FloorPlan206": 2224,
    "FloorPlan207": 2896,
    "FloorPlan208": 4544,
    "FloorPlan209": 5472,
    "FloorPlan210": 4720,
    "FloorPlan211": 2608,
    "FloorPlan212": 2368,
    "FloorPlan213": 5088,
    "FloorPlan214": 3440,
    "FloorPlan215": 6736,
    "FloorPlan216": 2784,
    "FloorPlan217": 2400,
    "FloorPlan218": 7680,
    "FloorPlan219": 3696,
    "FloorPlan220": 4128,
    "FloorPlan221": 2208,
    "FloorPlan222": 1680,
    "FloorPlan223": 3968,
    "FloorPlan224": 4976,
    "FloorPlan225": 2800,
    "FloorPlan226": 1696,
    "FloorPlan227": 4224,
    "FloorPlan228": 2752,
    "FloorPlan229": 3632,
    "FloorPlan230": 7488,
    "FloorPlan301": 1520,
    "FloorPlan302": 864,
    "FloorPlan303": 1408,
    "FloorPlan304": 2112,
    "FloorPlan305": 1568,
    "FloorPlan306": 1888,
    "FloorPlan307": 1712,
    "FloorPlan308": 1920,
    "FloorPlan309": 6336,
    "FloorPlan310": 1184,
    "FloorPlan311": 4256,
    "FloorPlan312": 1648,
    "FloorPlan313": 944,
    "FloorPlan314": 1376,
    "FloorPlan315": 1952,
    "FloorPlan316": 1008,
    "FloorPlan317": 2080,
    "FloorPlan318": 1984,
    "FloorPlan319": 1792,
    "FloorPlan320": 1152,
    "FloorPlan321": 1648,
    "FloorPlan322": 1984,
    "FloorPlan323": 3712,
    "FloorPlan324": 2048,
    "FloorPlan325": 3936,
    "FloorPlan326": 2272,
    "FloorPlan327": 1648,
    "FloorPlan328": 1280,
    "FloorPlan329": 1888,
    "FloorPlan330": 2432,
    "FloorPlan401": 1840,
    "FloorPlan402": 1440,
    "FloorPlan403": 1280,
    "FloorPlan404": 1072,
    "FloorPlan405": 624,
    "FloorPlan406": 1904,
    "FloorPlan407": 704,
    "FloorPlan408": 864,
    "FloorPlan409": 928,
    "FloorPlan410": 1616,
    "FloorPlan411": 1344,
    "FloorPlan412": 1168,
    "FloorPlan413": 1344,
    "FloorPlan414": 1008,
    "FloorPlan415": 1168,
    "FloorPlan416": 1584,
    "FloorPlan417": 1328,
    "FloorPlan418": 1184,
    "FloorPlan419": 656,
    "FloorPlan420": 528,
    "FloorPlan421": 656,
    "FloorPlan422": 960,
    "FloorPlan423": 1200,
    "FloorPlan424": 848,
    "FloorPlan425": 608,
    "FloorPlan426": 976,
    "FloorPlan427": 1040,
    "FloorPlan428": 1232,
    "FloorPlan429": 1392,
    "FloorPlan430": 1856}
