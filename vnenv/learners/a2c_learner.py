from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from .abs_learner import AbsLearner
from .returns_calc import _basic_return
from .loss_functions import _ac_loss
from vnenv.utils.convert import toNumpy, toTensor, dict2tensor


# to manage all the algorithm params
class A2CLearner(AbsLearner):

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        gamma: float,
        nsteps: int,
        vf_param: float = 0.5,
        ent_param: float = 0
    ) -> None:
        self.model = model
        self.dev = next(model.parameters()).device

        self.optimizer = optimizer
        self.nsteps = nsteps
        self.vf_param = vf_param
        self.gamma = np.float32(gamma)
        self.ent_param = ent_param

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

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, a, m = batched_exp['r'], batched_exp['a'], batched_exp['m']
        exp_num = r.shape[1]
        if rct == {}:
            obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
            model_out = self.model(dict2tensor(obs, dev=self.dev))
        else:
            model_out = self.rct_forward(obs, rct, m)
        # all data in model_out should in (batch_size, *)
        # reshape value to (exp_length+1, exp_num)
        v_array = toNumpy(model_out['value']).reshape(-1, exp_num)
        returns = _basic_return(
            v_array, r, m,
            self.gamma,
            self.nsteps
        )
        pi_loss, v_loss = _ac_loss(
            model_out['policy'][:-exp_num],
            model_out['value'][:-exp_num],
            toTensor(returns, self.dev),
            toTensor(a.reshape(-1, 1), self.dev),
            self.vf_param
        )
        obj_func = pi_loss + v_loss
        self.optimizer.zero_grad()
        obj_func.backward()
        self.optimizer.step()
        return dict(
            obj_func=obj_func.item(),
            pi_loss=pi_loss.item(),
            v_loss=v_loss.item()
        )
