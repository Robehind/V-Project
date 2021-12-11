import gym
from .abs_env import AbsEnv


class CartPolev1(AbsEnv):
    def __init__(
        self,
        dynamics_args={},
        event_args={'r_scale': 100.0},
        obs_args={'key': 'obs'},
        seed=None,
        train=True
    ) -> None:
        self._env = gym.make('CartPole-v1')
        if seed is not None:
            self._env.seed(seed)
        self.action_sz = self._env.action_space.n
        self.scale = event_args['r_scale']
        key = obs_args['key']
        self.keys = [key]
        self.shapes = {key: self._env.observation_space.shape}
        self.dtypes = {key: self._env.observation_space.dtype}

    def reset(self, *args, **kwargs):
        return {self.keys[0]: self._env.reset()}

    def step(self, action):
        o, r, d, info = self._env.step(action)
        return {self.keys[0]: o}, r/self.scale, d, info

    def update_settings(self, settings):
        pass

    def re_seed(self, seed):
        self._env.seed(seed)
