from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from .rct_learner import RCTLearner
from .returns_calc import _basic_return, _GAE
from methods.utils.convert import toNumpy, toTensor


# to manage all the algorithm params
class GradLearner(RCTLearner):

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim,
        optim_args: Dict,
        gamma: float,
        gae_lbd: float,
        vf_nsteps: int,
        vf_param: float = 0.5,
        vf_loss: str = 'mse_loss',
        ent_param: float = 0,
        grad_norm_max: float = 100.0,
        batch_loss_mean: bool = False
    ) -> None:
        self.model = model
        self.dev = next(model.parameters()).device
        self.optim = self.init_optim(optim, optim_args, self.dev)
        self.vf_nsteps = vf_nsteps
        self.gae_lbd = gae_lbd
        self.vf_param = vf_param
        self.vf_loss = getattr(F, vf_loss)
        self.gamma = gamma
        self.ent_param = ent_param
        self.grad_norm_max = grad_norm_max
        self.batch_loss_mean = batch_loss_mean
        self.reduction = 'mean' if batch_loss_mean else 'sum'

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, m = batched_exp['r'][:-1], batched_exp['m'][:-1]
        a = batched_exp['a'][:-1].reshape(-1, 1)
        exp_num = r.shape[1]
        exp_length = r.shape[0]
        # all data in model_out should be in (batch_size, *)
        model_out = self.model_forward(obs, rct, m)
        # reshape value to (exp_length+1, exp_num)
        v_array = toNumpy(model_out['value']).reshape(-1, exp_num)
        returns = _basic_return(
            v_array, r, m,
            self.gamma,
            self.vf_nsteps
        )
        if self.gae_lbd == 1.0 and exp_length <= self.vf_nsteps:
            adv = returns - v_array[:-1].reshape(-1, 1)
        else:
            adv = _GAE(v_array, r, m, self.gamma, self.gae_lbd)
        v_loss = self.vf_loss(
            model_out['value'][:-exp_num], toTensor(returns, self.dev),
            reduction=self.reduction)
        log_pi = F.log_softmax(model_out['policy'][:-exp_num], dim=1)
        pi = F.softmax(model_out['policy'][:-exp_num], dim=1)
        ent_loss = (- pi * log_pi).sum(1)
        log_pi_a = log_pi.gather(1, toTensor(a, self.dev))
        pi_loss = (-log_pi_a * toTensor(adv, dev=self.dev))
        if self.batch_loss_mean:
            ent_loss = ent_loss.mean()
            pi_loss = pi_loss.mean()
        else:
            ent_loss = ent_loss.sum()
            pi_loss = pi_loss.sum()
        # backward and optimize
        obj_func = \
            pi_loss + self.vf_param * v_loss - self.ent_param * ent_loss
        self.optim.zero_grad()
        obj_func.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optim.step()
        return dict(
            obj_func=obj_func.item(),
            pi_loss=pi_loss.item(),
            v_loss=v_loss.item(),
            ent_loss=ent_loss.item(),
        )
