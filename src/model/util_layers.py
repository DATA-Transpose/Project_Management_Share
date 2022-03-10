'''
@File    :   util_layers.py
@Time    :   2022/03/10 13:02:14
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import torch
import torch.nn as nn

class Align(nn.Module):
    """
    To align channel counts between blocks and layers
    """
    def __init__(self, c_in, c_out):
        """
        :param c_in: Number of input channels
        :param c_out: Number of out channels
        """
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, c_in, timestamp, n_vertex = x.shape
            x_align = torch.cat(
                [x, torch.zeros([batch_size, self.c_out - self.c_in, timestamp, n_vertex]).to(x)],
                dim=1
            )
        else:
            x_align = x

        return x_align
