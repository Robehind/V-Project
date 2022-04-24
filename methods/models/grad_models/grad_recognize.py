import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class MyLSTM(nn.Module):
    def __init__(
        self,
        in_sz,
        out_sz,
        learnable_x: bool = False,
        init: str = 'zeros'
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(in_sz, out_sz)
        self.rct_shapes = {'hx': (out_sz, ), 'cx': (out_sz, )}
        dtype = next(self.lstm.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype}
        init_f = getattr(torch, init)
        self.hx = Parameter(init_f(1, out_sz), learnable_x)
        self.cx = Parameter(init_f(1, out_sz), learnable_x)

    def forward(self, x, h):
        return self.lstm(x, h)
