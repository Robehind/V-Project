from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from .rct_learner import RCTLearner
from .returns_calc import _basic_return
from methods.utils.convert import toNumpy, toTensor, dict2tensor


# to manage all the algorithm params
class QLearner(RCTLearner):

    def __init__(
        self,
        model: nn.Module,
        optim: str,
        optim_args: Dict,
        gamma: float,
        nsteps: int,
        target_model: bool = False,
        sync_freq: int = 100,
        est_type: str = 'q_learning'  # TODO 'sarsa' 'e_sarsa' 'tree_backup'
    ) -> None:
        self.model = model
        # TODO Safe?
        self.target_model = deepcopy(model) if target_model else None
        self.update_cnt = 0
        self.sync_freq = sync_freq
        self.dev = next(model.parameters()).device

        self.optim = self.init_optim(optim, optim_args, self.dev)
        self.nsteps = nsteps
        self.gamma = np.float32(gamma)
        self.est_type = est_type

    def sync_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def target_forward(self, obs, rct):
        # no need for BPTT
        obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        rct = {k: v.reshape(-1, *v.shape[2:]) for k, v in rct.items()}
        with torch.no_grad():
            model_out = self.target_model(
                dict2tensor(obs, dev=self.dev),
                dict2tensor(rct, dev=self.dev)
            )
        return model_out['q_value']

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        obs, rct = batched_exp['obs'], batched_exp['rct']
        r, m = batched_exp['r'][:-1], batched_exp['m'][:-1]
        a = batched_exp['a']
        exp_num = r.shape[1]
        # all data in model_out should in (batch_size, *)
        model_out = self.model_forward(obs, rct, m)
        # gather q value
        _q_s = model_out['q_value'][:-exp_num]
        q_a = _q_s.gather(1, toTensor(a[:-1].reshape(-1, 1), self.dev))
        # q_s got by target model
        if self.target_model is None:
            q_s = model_out['q_value']
        else:
            q_s = self.target_forward(obs, rct)
        # TODO Q learning时，最后一个要gather最大的；Sarsa时要gather被选择的；
        # Expected sarsa要加权求和,以及树回溯
        if self.est_type == 'q_learning':
            q_t = q_s.max(dim=1)[0]
        elif self.est_type == 'sarsa':
            q_t = q_s.gather(1, toTensor(a.reshape(-1, 1), self.dev))
        else:
            raise NotImplementedError

        returns = _basic_return(
            toNumpy(q_t.reshape(-1, exp_num)), r, m,
            self.gamma,
            self.nsteps
        )
        q_loss = F.smooth_l1_loss(
            q_a, toTensor(returns, self.dev))
        obj_func = q_loss
        self.optim.zero_grad()
        obj_func.backward()
        self.optim.step()

        # sync target model
        if self.target_model is not None:
            self.update_cnt += 1
            self.update_cnt %= self.sync_freq
            if self.update_cnt == 0:
                self.sync_model()
        return dict(
            obj_func=obj_func.item(),
            q_loss=q_loss.item(),
        )
