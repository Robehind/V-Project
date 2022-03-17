from typing import Dict
from gym import Env


class TaskEnv(Env):
    """Base class for all Task Environment.
    """
    _tasks = None

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, t: Dict):
        self._tasks = t
        # maybe need more operation when modify task space

    def add_extra_info(self, *args, **kwargs):
        pass
