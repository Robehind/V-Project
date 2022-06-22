# Modified from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size,
        stride, dilation, padding, dropout
    ):
        super(TemporalBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        self.ll_conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)

        self.ll_conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.sigmoid = nn.Sigmoid()

    def net(self, x, block_num):
        x = self.ll_conv1(x)
        x = self.chomp1(x)
        x = F.leaky_relu(x)
        return x

    def init_weights(self):
        self.ll_conv1.weight.data.normal_(0, 0.01)
        self.ll_conv2.weight.data.normal_(0, 0.01)

    def forward(self, x, block_num):
        out = self.net(x, block_num)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            setattr(
                self,
                "ll_temporal_block{}".format(i),
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                ),
            )

    def forward(self, x):
        for i in range(self.num_levels):
            temporal_block = getattr(self, "ll_temporal_block{}".format(i))
            x = temporal_block(x, i)
        return x
