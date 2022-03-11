from typing import Any, Dict
import os
import torch
from vnenv.utils.net_utils import save_model


class AbsLearner:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        gamma: float,
        nsteps: int,
        *args,
        **kwargs
    ) -> None:
        pass

    def learn(
        self,
        batched_exps: Dict[str, Any]
    ) -> Dict[str, float]:
        return {'loss': 1}

    def checkpoint(self, path2save, steps):
        # save model
        title = self.model.__class__.__name__ + '_' + str(steps)
        save_model(self.model, path2save, title)
        optim_path = os.path.join(path2save, 'optim')
        # save optimizer
        optim_name = self.optimizer.__class__.__name__
        save_model(self.optimizer, optim_path,
                   f'{optim_name}_{steps}')
