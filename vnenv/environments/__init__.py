from gym.envs.registration import register

register(
    id='FcTdThor-v0',
    entry_point='vnenv.environments.ai2thor_env:FCTargetDrivenEnv',
)
