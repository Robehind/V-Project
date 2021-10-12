from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .a2c_learner import A2CLearner
from .returns_calc import _basic_return, _GAE
from vnenv.utils.convert import toNumpy, toTensor


# to manage all the algorithm params
class PPOLearner(A2CLearner):

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        pi_eps: float,
        repeat: int,
        recalc_adv: bool,
        clip_value: bool,
        gamma: float,
        gae_lbd: float,
        vf_nsteps: int,
        vf_param: float = 0.5,
        ent_param: float = 0
    ) -> None:
        super().__init__(
            model,
            optimizer,
            gamma,
            gae_lbd,
            vf_nsteps,
            vf_param,
            ent_param,
        )
        self.pi_eps = pi_eps
        self.repeat = repeat
        self.recalc_adv = recalc_adv
        self.clip_value = clip_value

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, m = batched_exp['r'][:-1], batched_exp['m'][:-1]
        a = batched_exp['a'][:-1].reshape(-1, 1)
        exp_num = r.shape[1]
        # pre calc old log pi_a
        with torch.no_grad():
            model_out = self.model_forward(obs, rct, m)
            ologpi = F.log_softmax(model_out['policy'][:-exp_num], dim=1)
            ologpi_a = ologpi.gather(1, toTensor(a, self.dev))
        adv = None

        loss_track = dict(
            obj_func=0,
            pi_loss=0,
            v_loss=0,
            ent_loss=0
        )

        for _ in range(self.repeat):
            model_out = self.model_forward(obs, rct, m)
            pi = F.softmax(model_out['policy'][:-exp_num], dim=1)
            pi_a = pi.gather(1, toTensor(a, self.dev))
            ratio = torch.exp(torch.log(pi_a) - ologpi_a)
            cliped = ratio.clamp(1-self.pi_eps, 1+self.pi_eps)
            v_array = toNumpy(model_out['value']).reshape(-1, exp_num)
            if adv is None or self.recalc_adv:
                adv = _GAE(v_array, r, m, self.gamma, self.gae_lbd)
                adv = toTensor(adv, self.dev)
            # loss construct
            pi_loss = - torch.min(ratio*adv, cliped*adv).mean()
            returns = _basic_return(
                v_array, r, m,
                self.gamma,
                self.vf_nsteps
            )
            if self.clip_value:
                # TODO
                raise NotImplementedError
            else:
                v_loss = F.smooth_l1_loss(
                    model_out['value'][:-exp_num], toTensor(returns, self.dev))
            ent_loss = (- torch.log(pi) * pi).sum(1).mean()
            obj_func = \
                pi_loss + self.vf_param * v_loss - self.ent_param * ent_loss
            self.optimizer.zero_grad()
            obj_func.backward()
            self.optimizer.step()
            loss_track['obj_func'] += obj_func.item()
            loss_track['pi_loss'] += pi_loss.item()
            loss_track['v_loss'] += v_loss.item()
            loss_track['ent_loss'] += ent_loss.item()

        # TODO why
        for k in loss_track:
            loss_track[k] /= self.repeat
        return loss_track
