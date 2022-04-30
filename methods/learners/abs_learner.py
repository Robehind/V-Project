from typing import Any, Dict
import os
import torch
from methods.utils.net_utils import save_model, optim2dev


class AbsLearner:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str,
        gamma: float,
        nsteps: int,
        *args,
        **kwargs
    ) -> None:
        pass

    def init_optim(self, optim, optim_args, dev):
        # get the model first
        optim_cls = getattr(torch.optim, optim)
        optimizer = optim_cls(
            self.model.parameters(), **optim_args)
        if 'load_optim_dir' in optim_args:
            path = optim_args['load_optim_dir']
            print("load optim %s" % path)
            optim.load_state_dict(torch.load(path))
            optim2dev(optim, dev)
        return optimizer

    def learn(
        self,
        batched_exps: Dict[str, Any]
    ) -> Dict[str, float]:
        return {'loss': 1}

    def checkpoint(self, path2save, steps):
        # save model
        title = self.model.__class__.__name__ + '_' + str(steps)
        save_model(self.model, path2save, title)
        # save optimizer
        optim_path = os.path.join(path2save, 'optim')
        optim_name = self.optim.__class__.__name__
        save_model(self.optim, optim_path,
                   optim_name, time_suffix=False)
