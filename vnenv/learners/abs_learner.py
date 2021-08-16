from typing import Any, Dict
import os
import torch
from vnenv.utils.net_utils import save_model
from vnenv.agents import AbsAgent


class AbsLearner:

    def __init__(
        self,
        agent: AbsAgent,
        optimizer: torch.optim,
        gamma: float,
        nsteps: int
    ) -> None:
        pass

    def learn(
        self,
        batched_exps: Dict[str, Any]
    ) -> Dict[str, float]:
        return NotImplemented

    def checkpoint(self, path2save, steps):
        optim_name = self.optimizer.__class__.__name__
        self.agent.save_model(path2save, steps)
        optim_path = os.path.join(path2save, 'optim')
        save_model(self.optimizer, optim_path,
                   f'{optim_name}_{steps}')
