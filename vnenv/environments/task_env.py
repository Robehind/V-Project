from gym import Env


class TaskEnv(Env):
    """TODO Task Environment"""
    _task_space = None

    @property
    def task_space(self):
        return self._task_space

    @task_space.setter
    def task_space(self, space):
        self._task_space = space
        # maybe need more operation when modify task space
