'''
@File    :   common_uilts.py
@Time    :   2022/03/10 18:48:49
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''


import os
import torch
import random
import warnings
import numpy as np

from logging import warning

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def try_gpu(i=0):
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        if gpu_count >= i + 1:
            return torch.device(f'cuda:{i}')
        else:
            warnings.warn("There are no enough devices, use the cuda:0 instead.")
            return torch.device(f'cuda:{0}')
    return torch.device('cpu')

def get_all_gpus() -> list:
    '''
    :return: list
    '''
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
