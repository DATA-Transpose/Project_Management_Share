'''
@File    :   output_layer.py
@Time    :   2022/03/10 15:21:20
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import torch
import torch.nn as nn
from src.model.temporal_layers import TemporalConvLayer

class OutputBlock(nn.Module):
    def __init__(
            self,
            ko,
            last_block_channel,
            channels,
            end_channel,
            n_vertex,
            gated_act_func,
            drop_rate
    ):
        super(OutputBlock, self).__init__()
        self.ko = ko
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.end_channel = end_channel
        self.n_vertex = n_vertex
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(ko, last_block_channel, channels[0], n_vertex, gated_act_func,
                                           self.enable_gated_act_func)
        self.fc1 = nn.Linear(channels[0], channels[1])
        self.fc2 = nn.Linear(channels[1], end_channel)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.act_func = 'sigmoid'
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_tc1 = self.tmp_conv1(x)
        x_tc1_ln = self.tc1_ln(x_tc1.permute(0, 2, 3, 1))
        x_fc1 = self.fc1(x_tc1_ln)
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

        return x_out
