from gym.envs.registration import register
from .task_env import TaskEnv


register(
    id='FcTdThor-v0',
    entry_point='taskenvs.ai2thor_env:FCTargetDrivenEnv',
)
__all__ = ['TaskEnv']
