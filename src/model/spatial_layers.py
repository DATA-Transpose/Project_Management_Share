'''
@File    :   spatial_layers.py
@Time    :   2022/03/10 14:14:54
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import torch
import torch.nn as nn
import torch.nn.init as init
from src.model.util_layers import Align

class ChebyshevConv(nn.Module):
    def __init__(self, c_in, c_out, ks, chebyshev_conv_matrix, enable_bias):
        super(ChebyshevConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.chebyshev_conv_matrix = chebyshev_conv_matrix
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(ks, c_in, c_out))
        if enable_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        init.xavier_uniform_(self.weight)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape
        x = x.reshape(n_vertex, -1)

        # According to the recurrence relation
        # T_0(L) = I
        # T_1(L) = L
        # T_n(L) = 2 * L * T_{n-1}(L) - T_{n-2}(L)
        # Then refer the chebyshev convolution formula: Formula.3
        x_0 = x
        x_1 = torch.mm(self.chebyshev_conv_matrix, x)
        x_list = []
        if self.ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.ks}.')
        elif self.ks - 1 == 0:
            x_list = [x_0]
        elif self.ks - 1 == 1:
            x_list = [x_0, x_1]
        elif self.ks - 1 >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.ks):
                x_list.append(torch.mm(2 * self.chebyshev_conv_matrix, x_list[k - 1]) - x_list[k - 2])
        x_tensor = torch.stack(x_list, dim=2)

        x_mul = torch.mm(x_tensor.view(-1, self.ks * c_in), self.weight.view(self.ks * c_in, -1)).view(-1, self.c_out)

        if self.bias is not None:
            x_chebyshev_conv = x_mul + self.bias
        else:
            x_chebyshev_conv = x_mul

        return x_chebyshev_conv

class GraphConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, graph_conv_type, conv_matrix):
        super(GraphConvLayer, self).__init__()
        self.ks = ks
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.graph_conv_type = graph_conv_type
        self.conv_matrix = conv_matrix
        self.enable_bias = True
        if self.graph_conv_type == 'chebyshev_conv':
            self.chebyshev_conv = ChebyshevConv(c_out, c_out, ks, conv_matrix, self.enable_bias)
        else:
            raise ValueError(f'ERROR: Just support chebyshev approximation currently')

    def forward(self, x):
        x_gc_in = self.align(x)
        batch_size, c_in, T, n_vertex = x_gc_in.shape
        if self.graph_conv_type == 'chebyshev_conv':
            x_gc = self.chebyshev_conv(x_gc_in)
        else:
            raise ValueError(f'ERROR: Just support chebyshev approximation currently')
        x_gc_with_rc = torch.add(x_gc.view(batch_size, self.c_out, T, n_vertex), x_gc_in)
        x_gc_out = x_gc_with_rc

        return x_gc_out
