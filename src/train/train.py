'''
@File    :   train.py
@Time    :   2022/03/10 22:38:42
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.train.earlystopping import EarlyStopping

def prepare_training_tools(
    learning_rate,
    weight_decay_rate,
    graph_conv_type,
    model_save_path,
    model, n_his,
    n_vertex,
    step_size,
    gamma,
    opt
):
    loss = nn.MSELoss()
    learning_rate = learning_rate
    weight_decay_rate = weight_decay_rate
    early_stopping = EarlyStopping(patience=30, path=model_save_path, verbose=True)

    if opt == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return loss, early_stopping, optimizer, scheduler


def train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter):
    def val(model, val_iter):
        model.eval()
        l_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_iter:
                y_pred = model(x).view(len(x), -1)
                l = loss(y_pred, y)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            return l_sum / n
    min_val_loss = np.inf
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        early_stopping(val_loss, model)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')

