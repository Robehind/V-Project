from taskenvs.tasker import Tasker
from gym.vector import VectorEnv
from typing import Optional, Dict
from .utils import get_scene_names
import random
from copy import deepcopy


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
            step = len(scenes)//n
            mod = len(scenes) % n
            for i in range(0, n*step, step):
                out.append(scenes[i:i + step])
            for i in range(0, mod):
                out[i].append(scenes[-(i+1)])

        task_list = [deepcopy(tasks) for _ in range(n)]
        for i, t in enumerate(task_list):
            t[scenes] = out[i]
        envs.call('set_tasks', task_list)
