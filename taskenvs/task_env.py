import random
from gym import Env
import numpy as np


class TaskEnv(Env):
    """TODO Task Environment"""
    _tasks = None

    def export_tasks(self):
        return self._tasks

    def set_tasks(self, t):
        self._tasks = t
        # maybe need more operation when modify task space

    def add_extra_info(self, *args, **kwargs):
        pass

    def seed(self, sd):
        random.seed(sd)
        np.random.seed(sd)
