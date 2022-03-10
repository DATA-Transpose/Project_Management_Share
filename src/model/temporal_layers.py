'''
@File    :   temporal_layers.py
@Time    :   2022/03/10 13:20:19
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.util_layers import Align


class CausalConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            enable_padding=False,
            dilation=1,
            groups=1,
            bias=True
    ):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, input_x):
        if self.__padding != 0:
            input_x = F.pad(input_x, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input_x)

        return result

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, n_vertex, act_func, enable_gated_act_func):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.act_func = act_func
        self.enable_gated_act_func = enable_gated_act_func
        self.align = Align(c_in, c_out)

        if enable_gated_act_func:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(self.kt, 1),
                                            enable_padding=False, dilation=nn.modules.utils._pair(1))
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(self.kt, 1),
                                            enable_padding=False, dilation=nn.modules.utils._pair(1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        if self.enable_gated_act_func:
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            # Temporal Convolution Layer (GLU)
            if self.act_func == 'glu':
                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x_glu = torch.mul((x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_glu

            # Temporal Convolution Layer (GTU)
            elif self.act_func == 'gtu':
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x_gtu = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_gtu
            else:
                raise ValueError(f'ERROR: activation function {self.act_func} is not defined.')
        else:
            # Currently, just go with GLU and GTU
            raise ValueError(f'ERROR: just support GLU and GTU currently.')

        return x_tc_out
