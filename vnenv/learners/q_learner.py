from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rct_learner import RCTLearner
from .returns_calc import _basic_return
from vnenv.utils.convert import toNumpy, toTensor, dict2tensor


# to manage all the algorithm params
class QLearner(RCTLearner):

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        gamma: float,
        nsteps: int,
        target_model: bool = False,  # TODO double learning
        est_type: str = 'q_learning'  # TODO 'sarsa' 'e_sarsa' 'tree_backup'
    ) -> None:
        self.model = model
        self.dev = next(model.parameters()).device
        self.target_model = target_model

        self.optimizer = optimizer
        self.nsteps = nsteps
        self.gamma = np.float32(gamma)

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, a, m = batched_exp['r'], batched_exp['a'], batched_exp['m']
        exp_num = r.shape[1]
        # TODO q_a got by target model
        # all data in model_out should in (batch_size, *)
        if rct == {}:
            obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
            model_out = self.model(dict2tensor(obs, dev=self.dev))
        else:
            model_out = self.rct_forward(obs, rct, m)
        # gather q value
        q_s = model_out['q_value'][:-exp_num]
        q_a = q_s.gather(1, toTensor(a.reshape(-1, 1), self.dev))
        # TODO Q learning时，最后一个要gather最大的；Sarsa时要gather被选择的；
        # Expected sarsa要加权求和,以及树回溯
        q_last = model_out['q_value'][-exp_num:]
        last_qa = q_last.max(dim=1)[0]

        returns = _basic_return(
            toNumpy(q_a.reshape(-1, exp_num)), toNumpy(last_qa), r, m,
            self.gamma,
            self.nsteps
        )
        q_loss = F.smooth_l1_loss(
            q_a, toTensor(returns, self.dev))
        obj_func = q_loss
        self.optimizer.zero_grad()
        obj_func.backward()
        self.optimizer.step()
        return dict(
            obj_func=obj_func.item(),
            q_loss=q_loss.item(),
        )
