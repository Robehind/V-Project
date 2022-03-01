from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from .abs_agent import AbsAgent
from gym.vector import VectorEnv
from vnenv.utils.convert import dict2tensor, dict2numpy
import vnenv.agents.select_funcs as sfcs


class BaseAgent(AbsAgent):
    """单模型智能体。管理将变成输入的模型输出（例如LSTM的隐藏状态）\
       仅用于采样。
    """
    def __init__(
        self,
        model: nn.Module,
        Venv: VectorEnv,  # 传环境进来只是为了获取一些参数
        gpu_ids: Optional[List[int]],
        select_func: str,
        select_params: List[Any] = [],  # TODO 暂时只有epsilon需要传,可以是generator
    ):
        self.gpu_ids = gpu_ids
        self.model = model
        self.proc_num = Venv.num_envs
        # TODO 多gpu训练同一个模型, DataParallel
        if gpu_ids is not None:
            self.model = self.model.cuda(gpu_ids[0])
        self.dev = next(self.model.parameters()).device
        # if model has recurrent inputs
        self.rct_shapes = {}
        self.rct_dtypes = {}
        self.rct = {}  # values in Tensor
        # TODO learnable rct
        if hasattr(model, 'rct_shapes'):
            self.rct_shapes = model.rct_shapes
            self.rct_dtypes = model.rct_dtypes
            self.rct = {
                k: torch.zeros((self.proc_num, *v),
                               dtype=self.rct_dtypes[k], device=self.dev)
                for k, v in self.rct_shapes.items()
            }
        self.select = getattr(sfcs, select_func)
        self.select_params = select_params

    def action(
        self,
        obs: Dict[str, np.ndarray],
        last_done: np.ndarray
    ) -> Tuple[List[int], Dict[str, np.ndarray]]:
        # reset rct
        self._reset_rct(last_done == 1)
        last_rct = self.get_rct()
        # no grad when sampling actions
        with torch.no_grad():
            # TODO copy?
            out = self.model(
                dict2tensor(obs.copy(), self.dev),
                self.rct
            )
        # if has recurrent states, update
        if self.rct_shapes != {}:
            del self.rct
            self.rct = out['rct']
        # action selection
        return self.select(out, *self.select_params), last_rct

    def _reset_rct(self, idx: np.ndarray):
        # reset recurrent data specified by idx to 0
        # TODO learnable init state?
        assert not isinstance(idx, bool)
        for v in self.rct.values():
            v[idx] = 0

    def get_rct(self) -> Dict[str, np.ndarray]:
        return dict2numpy(self.rct)
