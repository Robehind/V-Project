from typing import Dict
import torch
from .abs_learner import AbsLearner
from .returns_calc import _basic_return
from .loss_functions import _ac_loss
from vnenv.agents import AbsAgent
from vnenv.utils.convert import toNumpy, toTensor
import numpy as np


# to manage all the algorithm params
class A2CLearner(AbsLearner):

    def __init__(
        self,
        agent: AbsAgent,
        optimizer: torch.optim,
        gamma: float,
        nsteps: int,
        vf_param: float = 0.5,
        ent_param: float = 0
    ) -> None:
        self.agent = agent
        self.proc_num = agent.proc_num
        self.dev = agent.device

        self.optimizer = optimizer
        self.nsteps = nsteps
        self.vf_param = vf_param
        self.gamma = gamma
        self.ent_param = ent_param

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, r = batched_exp['o'], batched_exp['r']
        a, m = batched_exp['a'], batched_exp['m']
        model_out = self.agent.model_forward(obs)
        # reshape to (sample_steps+1, env_num)
        v_array = toNumpy(model_out['value']).reshape(-1, self.proc_num)
        returns = _basic_return(
            v_array, r, m,
            self.gamma, self.nsteps
        )
        pi_loss, v_loss = _ac_loss(
            model_out['policy'][:-self.proc_num],
            model_out['value'][:-self.proc_num],
            toTensor(returns, self.dev),
            toTensor(a, self.dev),
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
