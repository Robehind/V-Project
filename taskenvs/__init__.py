from gym.envs.registration import register

register(
    id='FcTdThor-v0',
    entry_point='taskenvs.ai2thor_env:FCTargetDrivenEnv',
)
