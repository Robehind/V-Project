from typing import Dict
import numpy as np
import torch
from .abs_learner import AbsLearner
from vnenv.utils.convert import toTensor, dict2tensor


class RCTLearner(AbsLearner):

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

    def rct_forward(
        self,
        obs: Dict[str, np.ndarray],
        rct: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        exp_length = mask.shape[0]
        # model forward step by step
        rct_t = {k: toTensor(v[0], self.dev) for k, v in rct.items()}
        output = {}  # {'rct': rct_t}
        for i in range(exp_length+1):
            obs_t = {k: v[i] for k, v in obs.items()}
            out = self.model(
                dict2tensor(obs_t, self.dev),
                rct_t,
            )
            rct_t = out.pop('rct')

            for k in out:
                output[k] = torch.cat([output[k], out[k]]) \
                    if k in output else out[k]
            if i < exp_length-1:
                self.reset_rct(rct_t, mask[i] == 1)
        return output

    def reset_rct(
        self,
        rct,
        idxes
    ):
        # detach part of the batch to stop gradient between epis
        # TODO more elegant detach method?
        for k, v in rct.items():
            nv = torch.zeros_like(v)
            nv[idxes] = v[idxes]
            rct[k] = nv
