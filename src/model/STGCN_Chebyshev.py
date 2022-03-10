'''
@File    :   STGCN_Chebyshev.py
@Time    :   2022/03/10 18:36:30
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''


import torch.nn as nn

from src.model.st_block import STConvBlock
from src.model.output_layer import OutputBlock


class STGCNChebyshev(nn.Module):
    def __init__(self, kt, ks, blocks, t, n_vertex, gated_act_func, graph_conv_type, chebyshev_conv_matrix, drop_rate):
        super(STGCNChebyshev, self).__init__()
        modules = []
        for layer_index in range(len(blocks) - 3):
            modules.append(
                STConvBlock(
                    kt,
                    ks,
                    n_vertex,
                    blocks[layer_index][-1],
                    blocks[layer_index + 1],
                    gated_act_func,
                    graph_conv_type,
                    chebyshev_conv_matrix,
                    drop_rate
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = t - (len(blocks) - 3) * 2 * (kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, gated_act_func, drop_rate
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'leaky_relu':
                x_act_func = self.leaky_relu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            else:
                raise ValueError(
                    f'ERROR: The {self.act_func} activate function is not supported currently.')
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            x_out = x_fc2
        else:
            raise ValueError('Error: Invalid ko.')

        return x_out

