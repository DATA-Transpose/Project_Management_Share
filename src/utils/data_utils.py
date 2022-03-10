'''
@File    :   data_utils.py
@Time    :   2022/03/10 22:48:41
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import math
import torch
import numpy as np
import pandas as pd
import torch.utils as utils

from sklearn import preprocessing

def data_composer(
        file_path,
        n_his,
        n_pred,
        day_slot,
        batch_size,
        device,
        val_set_rate=0.15,
        test_set_rate=0.15,
        channel_size=1
):
    """
    :param file_path: Signal data file path.
    :param n_his: How many records will be referred as history.
    :param n_pred: How many steps after the last history record will be generated as prediction.
    :param day_slot: How many slots will a day be split. (288 slots if take 5 minutes as interval)
    :param batch_size: batch size.
    :param device: cpu or gpu: torch.device.
    :param val_set_rate: Rate of validation set among whole data.
    :param test_set_rate: Rate of test set among whole data.
    :param channel_size: Channel size.
    :return: Tensors of x and y for each sets, which can be used for torch model.
    """
    train, val, test = data_load(file_path, val_set_rate, test_set_rate)

    # train, train_zp = z_score(train)
    # val, val_zp = z_score(val)
    # test, test_zp = z_score(test)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    # z_params = {
    #     'train': train_zp,
    #     'val': val_zp,
    #     'test': test_zp
    # }

    x_train, y_train = data_transform(train, n_his, n_pred, device, channel_size)
    x_val, y_val = data_transform(val, n_his, n_pred, device, channel_size)
    x_test, y_test = data_transform(test, n_his, n_pred, device, channel_size)


    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return zscore, train_iter, val_iter, test_iter


def data_load(file_path, val_set_rate=0.15, test_set_rate=0.15):
    """
    Load data from file and split it into train, val and test.
    :param file_path: Signal data file path.
    :param val_set_rate: Rate of validation set among whole data.
    :param test_set_rate: Rate of test set among whole data.
    :return: train, val, test sets in pd.DataFrame
    """
    try:
        df = pd.read_csv(file_path, header=None).values

        data_col = df.shape[0]
        len_val = int(math.floor(data_col * val_set_rate))
        len_test = int(math.floor(data_col * test_set_rate))
        len_train = int(data_col - len_val - len_test)

        train = df[:len_train]
        val = df[len_train:len_train + len_val]
        test = df[len_train + len_val:]

        return train, val, test

    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')


def data_transform(data, n_his, n_pred, device, channel_size=1):
    """
    Split a data set into x sequence and y sequence
    :param data: data set
    :param n_his: How many records will be referred as history.
    :param n_pred: How many steps after the last history record will be generated as prediction.
    :param device: cpu or gpu: torch.device
    :param channel_size: Channel size
    :return:
    """
    len_node = data.shape[1]
    len_record = len(data)

    # Each data set could generate num groups,
    # which the group is combined with x(history) sequence and y(prediction) sequence.
    num = len_record - n_his - n_pred

    # To organize data by torch requirement [batch_size, channel_size, height, width]
    x = np.zeros([num, channel_size, n_his, len_node])
    y = np.zeros([num, len_node])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(channel_size, n_his, len_node)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
