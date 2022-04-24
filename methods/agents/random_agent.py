from typing import Dict, List, Tuple
import numpy as np
from .abs_agent import AbsAgent
import random
from gym.vector import VectorEnv


class RandomAgent(AbsAgent):
    """随机动作智能体
    """
    def __init__(
        self,
        model,
        env: VectorEnv,
        *args,
        **kwargs
    ):
        self.action_sz = env.single_action_space.n
        self.proc_num = env.num_envs
        self.model = model
        self.rct_shapes = {}
        self.rct_dtypes = {}

    def action(
        self,
        obs: Dict[str, np.ndarray] = None,
        done=None
    ) -> Tuple[List[str], np.ndarray]:
        a_ids = list(range(self.action_sz))
        a_idx = np.array([random.choice(a_ids) for _ in range(self.proc_num)])
        return a_idx, {}
