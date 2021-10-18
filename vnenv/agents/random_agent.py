from typing import Dict, List, Tuple
import numpy as np
from .abs_agent import AbsAgent
import random


class RandomAgent(AbsAgent):
    """随机动作智能体
    """
    def __init__(
        self,
        model,
        env,
        *args,
        **kwargs
    ):
        self.action_sz = env.action_sz
        self.proc_num = env.env_num

    def action(
        self,
        obs: Dict[str, np.ndarray] = None,
        done=None
    ) -> Tuple[List[str], np.ndarray]:
        a_ids = list(range(self.action_sz))
        a_idx = np.array([random.choice(a_ids) for _ in range(self.proc_num)])
        return a_idx, {}
