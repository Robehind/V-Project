import functools
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from .rct_learner import RCTLearner
from .returns_calc import _basic_return, _GAE
from methods.utils.net_utils import optim2dev
from methods.utils.convert import toNumpy, toTensor, dict2tensor
from functorch import grad, make_functional_with_buffers, vmap


# to manage all the algorithm params
class SAVNLearner(RCTLearner):

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim,
        optim_args: Dict,
        inner_lr: float,
        max_iter: int,
        meta_steps: int,
        gamma: float,
        gae_lbd: float,
        vf_nsteps: int,
        vf_param: float = 0.5,
        vf_loss: str = 'mse_loss',
        ent_param: float = 0,
        grad_norm_max: float = 100.0,
        batch_loss_mean: bool = False
    ) -> None:
        self.inner_lr = inner_lr
        self.meta_steps = meta_steps
        self.max_iter = max_iter
        self.model = model
        self.dev = next(model.parameters()).device
        self.sp_main = make_functional_with_buffers(self.model)
        self.sp_meta = make_functional_with_buffers(self.model.ll_tc)
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

    def init_optim(self, optim, optim_args, dev):
        # get the model first
        optim_cls = getattr(torch.optim, optim)
        optimizer = optim_cls(
            [self.sp_main[1], self.sp_meta[1]], **optim_args)
        if 'load_optim_dir' in optim_args:
            path = optim_args['load_optim_dir']
            print("load optim %s" % path)
            optim.load_state_dict(torch.load(path))
            optim2dev(optim, dev)
        return optimizer

    def loss_for_task(
        self,
        main,
        meta,
        single_exp,
    ):
        mainf, mainp, mainb = main
        metaf, metap, metab = meta
        lr = self.inner_lr
        # single exp : (exp_length, *data_shape)
        obs, rct = single_exp['obs'], single_exp['rct']
        r, m = single_exp['r'][:-1], single_exp['m'][:-1]
        a = single_exp['a'][:-1].reshape(-1, 1)
        exp_length = r.shape[0]
        n_iter = min(exp_length // self.meta_steps, self.max_iter)

        new_params = mainp
        rct_t = {
            k: toTensor(v[0].unsqueeze(0), self.dev)
            for k, v in rct.items()}
        for i in range(n_iter):
            learned_input = []
            for i in range(self.meta_steps*i, self.meta_steps*i):
                obs_t = {k: v[i].unsqueeze(0) for k, v in obs.items()}
                out = mainf(
                    new_params, mainb,
                    dict2tensor(obs_t, self.dev), rct_t)
                rct_t = out.pop('rct')
                tmp = torch.cat(
                    [rct_t['hx'], rct_t['action_prob']], dim=1)
                learned_input.append(tmp)
                if i < exp_length-1:
                    self.reset_rct(rct_t, m[i] == 1)
            learned_input = torch.cat(learned_input, dim=0)
            grads = grad(metaf)(metap, metab, learned_input)
            new_params = [p - g * lr for p, g in zip(new_params, grads)]

        # learning
        rct_t = {
            k: toTensor(v[0].unsqueeze(0), self.dev)
            for k, v in rct.items()}
        model_out = {}
        for i in range(exp_length+1):
            obs_t = {k: v[i].unsqueeze(0) for k, v in obs.items()}
            out = mainf(
                new_params, mainb,
                dict2tensor(obs_t, self.dev), rct_t)
            rct_t = out.pop('rct')

            for k in out:
                model_out[k] = torch.cat([model_out[k], out[k]]) \
                    if k in model_out else out[k]
            if i < exp_length-1:
                self.reset_rct(rct_t, m[i] == 1)
        # reshape value to (exp_length+1, exp_num)
        v_array = toNumpy(model_out['value']).reshape(-1, 1)
        returns = _basic_return(
            v_array, r, m,
            self.gamma, self.vf_nsteps)
        if self.gae_lbd == 1.0 and exp_length <= self.vf_nsteps:
            adv = returns - v_array[:-1].reshape(-1, 1)
        else:
            adv = _GAE(v_array, r, m, self.gamma, self.gae_lbd)
        v_loss = self.vf_loss(
            model_out['value'][:-1], toTensor(returns, self.dev),
            reduction=self.reduction)
        log_pi = F.log_softmax(model_out['policy'][:-1], dim=1)
        pi = F.softmax(model_out['policy'][:-1], dim=1)
        ent_loss = (- pi * log_pi).sum(1)
        log_pi_a = log_pi.gather(1, toTensor(a, self.dev))
        pi_loss = (-log_pi_a * toTensor(adv, dev=self.dev))
        if self.batch_loss_mean:
            ent_loss = ent_loss.mean()
            pi_loss = pi_loss.mean()
        else:
            ent_loss = ent_loss.sum()
            pi_loss = pi_loss.sum()
        obj_func = \
            pi_loss + self.vf_param * v_loss - self.ent_param * ent_loss
        return obj_func, pi_loss.item(), v_loss.item(), ent_loss.item()

    def copy_params(self, mainp, metap):
        for pf, pt in zip(mainp, self.model.parameters()):
            pt.data = pf.data
        for pf, pt in zip(metap, self.model.ll_tc.parameters()):
            pt.data = pf.data

    def learn(
        self,
        batched_exp: Dict[str, np.ndarray]
    ) -> Dict[str, float]:

        loss_calc = functools.partial(
            self.learn, self.sp_main, self.sp_meta)
        obj_func, pi_loss, v_loss, ent_loss = vmap(
            loss_calc, in_dims=1)(batched_exp)

        # Compute the maml loss by summing together the returned losses.
        self.optim.zero_grad()
        obj_func.sum().backward()
        clip_grad_norm_([self.sp_main[1], self.sp_meta[1]], self.grad_norm_max)
        self.optim.step()
        self.copy_params(self.sp_main[1], self.sp_meta[1])
        return dict(
            obj_func=obj_func.sum().item(),
            pi_loss=pi_loss.sum().item(),
            v_loss=v_loss.sum().item(),
            ent_loss=ent_loss.sum().item())
