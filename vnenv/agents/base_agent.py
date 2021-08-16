from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .abs_agent import AbsAgent
from vnenv.utils.convert import toTensor
from vnenv.utils.net_utils import save_model
from vnenv.environments import VecEnv


class BaseAgent(AbsAgent):
    """单模型智能体。帮助模型完成数据转换并管理将变成输入的模型输出（例如LSTM的隐藏状态）
    """
    def __init__(
        self,
        model: nn.Module,
        Venv: VecEnv,  # 传环境进来只是为了获取一些参数
        gpu_ids: Optional[List[int]] = None
    ):
        self.gpu_ids = gpu_ids
        self.model = model
        self.done = False
        self.proc_num = Venv.env_num
        # TODO 多gpu训练同一个模型,DataParallel
        if gpu_ids is not None:
            self.model = self.model.cuda(gpu_ids[0])
        self.device = next(self.model.parameters()).device

    def model_forward(
        self,
        obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """obs is dict. values of obs must in numpy,
           and first dim is batch dim"""
        model_input = obs.copy()  # 防止obs被改变，因为obs在外部还被保存了一次
        for k in model_input:
            model_input[k] = toTensor(
                model_input[k], self.device, dtype=torch.float32
            )
        return self.model.forward(model_input)

    def action(
        self,
        obs: Dict[str, np.ndarray],
        done: Optional[np.ndarray] = None
    ) -> List[int]:
        # 调用action的时候代表是在采样，不计算梯度
        with torch.no_grad():
            out = self.model_forward(obs)
        pi = out['policy']
        # softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim=1).cpu()
        # 采样
        action_idx = prob.multinomial(1).numpy().squeeze(1)
        return action_idx

    def save_model(self, path2save, steps):
        title = self.model.__class__.__name__ + '_' + str(steps)
        save_model(self.model, path2save, title)
