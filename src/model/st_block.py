'''
@File    :   st_block.py
@Time    :   2022/03/10 18:33:20
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import torch
import torch.nn as nn

from src.model.temporal_layers import TemporalConvLayer
from src.model.spatial_layers import GraphConvLayer


class STConvBlock(nn.Module):
    def __init__(
            self,
            kt,
            ks,
            n_vertex,
            last_block_channel,
            channels,
            gated_act_func,
            graph_conv_type,
            graph_conv_matrix,
            drop_rate
    ):
        super(STConvBlock, self).__init__()
        self.kt = kt
        self.ks = ks
        self.n_vertex = n_vertex
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.graph_conv_act_func = 'relu'
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(kt, last_block_channel, channels[0], n_vertex, gated_act_func,
                                           self.enable_gated_act_func)
        self.graph_conv = GraphConvLayer(ks, channels[0], channels[1], graph_conv_type, graph_conv_matrix)
        self.tmp_conv2 = TemporalConvLayer(kt, channels[1], channels[2], n_vertex, gated_act_func,
                                           self.enable_gated_act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)  # The 1-st temporal layer
        x_graph_conv = self.graph_conv(x_tmp_conv1)  # The spatial layer
        if self.graph_conv_act_func == 'sigmoid':
            x_act_func = self.sigmoid(x_graph_conv)
        elif self.graph_conv_act_func == 'tanh':
            x_act_func = self.tanh(x_graph_conv)
        elif self.graph_conv_act_func == 'relu':
            x_act_func = self.relu(x_graph_conv)
        elif self.graph_conv_act_func == 'leaky_relu':
            x_act_func = self.leaky_relu(x_graph_conv)
        elif self.graph_conv_act_func == 'elu':
            x_act_func = self.elu(x_graph_conv)
        else:
            raise ValueError(
                f'ERROR: The {self.graph_conv_act_func} activate function is not supported currently.')
        x_tmp_conv2 = self.tmp_conv2(x_act_func)  # The 2-ed temporal layer
        x_tc2_ln = self.tc2_ln(x_tmp_conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_do = self.do(x_tc2_ln)
        x_st_conv_out = x_do

        return x_st_conv_out
