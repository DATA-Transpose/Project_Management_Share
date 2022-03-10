'''
@File    :   visualization.py
@Time    :   2022/03/10 19:36:14
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import matplotlib.pyplot as plt

def plot_seq(name, seq):
    plt.figure(figsize=(20, 4))
    plt.plot(seq)
    # plt.show()
    plt.savefig(f'./runs/tmp/plot_seq_{name}.jpg')


def plot_batch_seq(data):
    batch_num = data.shape[1]
    plt.figure(figsize=(20, batch_num))

    for i in range(batch_num):
        plt.subplot(batch_num, 1, i + 1)
        plt.plot(data[:, i])
    # plt.show()
    plt.savefig('./runs/tmp/plot_batch_seq.jpg')

def plot_acc(name, y_list, pre_y):
    plt.figure(figsize=(20, 4))
    plt.plot(y_list, label='Actual')
    plt.plot(pre_y, label='Prediction')
    # plt.show()
    plt.savefig(f'./runs/tmp/plot_acc_{name}.jpg')