from gym import Env


class TaskEnv(Env):
    """Base class for all Task Environment.
    """
    _tasks = None

    def export_tasks(self):
        return self._tasks

    def set_tasks(self, t):
        self._tasks = t
        # maybe need more operation when modify task space

    def add_extra_info(self, *args, **kwargs):
        pass
