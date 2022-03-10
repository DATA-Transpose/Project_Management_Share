'''
@File    :   main.py
@Time    :   2022/03/10 18:42:56
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import os
import torch
import argparse
import numpy as np
import pandas as pd

import src.utils.graph_utils as GU
import src.utils.common_uilts as CU

from src.train.eval import test
from src.train.train import train, prepare_training_tools
from src.utils.data_utils import data_composer
from src.utils.visualization import plot_seq, plot_acc
from src.model.STGCN_Chebyshev import STGCNChebyshev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGCN.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--cuda', type=int, default=0, help='Index of cuda device.')
    parser.add_argument('--history_window', type=int, default=18, help='History window size.')
    parser.add_argument('--predict_window', type=int, default=18, help='Prediction window size.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout.')
    parser.add_argument('--epochs', type=int, default=500, help='epochs.')
    parser.add_argument('--spacial_kernel', type=int, default=3, help='Spacial kernel size.')
    parser.add_argument('--tempora_kernel', type=int, default=3, help='Temporal kernel size.')
    parser.add_argument('--day_slot', type=int, default=288, help='Day slot.')
    parser.add_argument('--vertex', type=int, default=228, help='Vertex number.')
    parser.add_argument('--model_path', type=str, default='./runs/models/pemsd7-m/', help='Model saving path.')
    parser.add_argument('--time_intvl', type=int, default=5, help='Model saving path.')
    parser.add_argument('--data_set', type=str, default='PeMS-M', help='Dataset.')

    args = parser.parse_args()
    print(args)

    CU.set_seed(args.seed)
    device = CU.try_gpu(args.cuda)
    print(device)
    CU.get_all_gpus()

    stblock_num = 2
    Ko = args.history_window - (args.tempora_kernel - 1) * 2 * stblock_num
    time_pred = args.predict_window * args.time_intvl
    time_pred_str = str(time_pred) + '_mins'
    model_save_path = f'{args.model_path}{time_pred_str}'
    log_name = []
    for item in vars(args):
        if item in ['cuda', 'model_path']:
            continue
        log_name.append(item)
        log_name.append(str(getattr(args, item)))
    log_name = '_'.join(log_name)
    model_save_path = f'{model_save_path}_{log_name}.pth'
    print(model_save_path)

    graph_conv_type = 'chebyshev_conv'
    gated_act_func = 'glu'
    learning_rate = 0.001
    weight_decay_rate = 0.0005
    step_size = 10
    gamma = 0.999

    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    pwd = os.path.abspath('.')
    if args.data_set == 'PeMS-M':
        origin_w_path = os.path.join(pwd, 'data/PeMS-M/W_228.csv')
        origin_v_path = os.path.join(pwd, 'data/PeMS-M/V_228.csv')
        plot_data = pd.read_csv(origin_v_path).values
        plot_seq(log_name, plot_data[:, 0])
        W = GU.weight_matrix(origin_w_path, 0.747, 0.05)
    if args.data_set == 'Brisbane':
        W = np.load(os.path.join(pwd, 'data/Brisbane/city_w_adj.npy'))
        origin_v_path = os.path.join(pwd, 'data/Brisbane/city_traffic_signal.csv')

    scaled_normalized_L = GU.scaled_laplacian(W)
    chebyshev_conv_matrix = torch.from_numpy(scaled_normalized_L).float().to(device)
    stgcn_chebconv = STGCNChebyshev(
        args.tempora_kernel,
        args.spacial_kernel,
        blocks,
        args.history_window,
        args.vertex,
        gated_act_func,
        graph_conv_type,
        chebyshev_conv_matrix,
        args.dropout
    ).to(device)
    model = stgcn_chebconv

    loss, early_stopping, optimizer, scheduler = prepare_training_tools(
        learning_rate, weight_decay_rate, graph_conv_type, model_save_path, model, args.history_window, args.vertex, step_size, gamma, 'AdamW'
    )

    [
        z_params,
        train_iter,
        val_iter,
        test_iter
    ] = data_composer(origin_v_path, args.history_window, args.predict_window, args.day_slot, args.batch_size, device, 0.15, 0.15, 1)
    
    train(loss, args.epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter)

    # visualize_model(model, train_iter)
    y_list, pre_y = test(z_params, loss, model, test_iter, model_save_path)

    y_list = y_list[:, 0]
    pre_y = pre_y[:, 0]

    plot_acc(log_name, y_list, pre_y)
