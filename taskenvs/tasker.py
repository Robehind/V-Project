from gym.vector import VectorEnv
from typing import Dict, Optional


class Tasker:
    def __init__(
        self,
        envs: VectorEnv,
        tasks: Optional[Dict] = None,
        *args,
        **kwargs
    ) -> None:
        if tasks is None:
            return
        envs.set_attr('tasks', tasks)

    def next_tasks(
        self,
        update_steps,
        *args,
        **kwargs
    ):
        return False
