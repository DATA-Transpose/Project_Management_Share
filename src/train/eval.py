'''
@File    :   test.py
@Time    :   2022/03/10 22:43:18
@Author  :   ChildEden 
@Version :   1.0
@Contact :   chenhao.zhang@uq.edu.au
@Desc    :   None
'''

import torch
import numpy as np

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n

        return mse


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    y_list = np.array([])
    pre_y = np.array([])
    idx = 0
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy())
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy())

            if idx == 0:
                y_list = y
                pre_y = y_pred
            else:
                y_list = np.vstack((y_list, y))
                pre_y = np.vstack((pre_y, y_pred))
            idx += 1

            y = y.reshape(-1)
            y_pred = y_pred.reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        # return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE, y_list, pre_y


def test(zscore, loss, model, test_iter, model_save_path):
    best_model = model
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = evaluate_model(best_model, loss, test_iter)
    print('Test loss {:.6f}'.format(test_MSE))
    # test_MAE, test_MAPE, test_RMSE = utility.evaluate_metric(best_model, test_iter, zscore)
    # print(f'MAE {test_MAE:.6f} | MAPE {test_MAPE:.8f} | RMSE {test_RMSE:.6f}')
    test_MAE, test_RMSE, test_WMAPE, y_list, pre_y = evaluate_metric(best_model, test_iter, zscore)
    print(f'MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')
    return y_list, pre_y
