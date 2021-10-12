from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rct_learner import RCTLearner
from .returns_calc import _basic_return, _GAE
from vnenv.utils.convert import toNumpy, toTensor


# to manage all the algorithm params
class A2CLearner(RCTLearner):

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        gamma: float,
        gae_lbd: float,
        vf_nsteps: int,
        vf_param: float = 0.5,
        ent_param: float = 0
    ) -> None:
        self.model = model
        self.dev = next(model.parameters()).device

        self.optimizer = optimizer
        self.vf_nsteps = vf_nsteps
        self.gae_lbd = gae_lbd
        self.vf_param = vf_param
        self.gamma = gamma
        self.ent_param = ent_param

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, m = batched_exp['r'][:-1], batched_exp['m'][:-1]
        a = batched_exp['a'][:-1].reshape(-1, 1)
        exp_num = r.shape[1]
        # all data in model_out should in (batch_size, *)
        model_out = self.model_forward(obs, rct, m)
        # reshape value to (exp_length+1, exp_num)
        v_array = toNumpy(model_out['value']).reshape(-1, exp_num)
        adv = _GAE(v_array, r, m, self.gamma, self.gae_lbd)
        returns = _basic_return(
            v_array, r, m,
            self.gamma,
            self.vf_nsteps
        )
        v_loss = F.smooth_l1_loss(
            model_out['value'][:-exp_num], toTensor(returns, self.dev))
        pi = F.softmax(model_out['policy'][:-exp_num], dim=1)
        ent_loss = (- torch.log(pi) * pi).sum(1).mean()
        pi_a = pi.gather(1, toTensor(a, self.dev))
        pi_loss = (-torch.log(pi_a) * toTensor(adv, dev=self.dev)).mean()
        obj_func = \
            pi_loss + self.vf_param * v_loss - self.ent_param * ent_loss
        self.optimizer.zero_grad()
        obj_func.backward()
        self.optimizer.step()
        return dict(
            obj_func=obj_func.item(),
            pi_loss=pi_loss.item(),
            v_loss=v_loss.item(),
            ent_loss=ent_loss.item()
        )
