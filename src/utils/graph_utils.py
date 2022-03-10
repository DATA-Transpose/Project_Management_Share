import time

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    """
    try:
        adj_matrix = pd.read_csv(file_path, header=None).values

        # check whether W is a 0/1 matrix.
        if set(np.unique(adj_matrix)) == {0, 1}:
            print('The input graph is a 0/1 matrix; set "scaling" to False.')
            scaling = False

        if scaling:
            n = adj_matrix.shape[0]
            adj_matrix = adj_matrix / 10000.
            adj_m_2, mask = adj_matrix * adj_matrix, np.ones([n, n]) - np.identity(n)
            # refer to Eq.10
            # return np.exp(-adj_m_2 / sigma2) * (np.exp(-adj_m_2 / sigma2) >= epsilon) * mask
            return np.exp(-adj_m_2 / sigma2) * (np.exp(-adj_m_2 / sigma2) >= epsilon)
        else:
            return adj_matrix
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')


def scaled_laplacian(adj_matrix):
    """
    Normalized graph Laplacian function.
    :param adj_matrix: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    # D ->  diagonal degree matrix
    n, D = np.shape(adj_matrix)[0], np.diag(np.sum(adj_matrix, axis=1))
    # L -> graph Laplacian
    L = D - adj_matrix

    # L_normalized = D^{-0.5} * L * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
    D_inv_sqrt = fractional_matrix_power(D, -0.5)
    normalized_L = np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt)

    lambda_max = max(eigvalsh(normalized_L))

    return np.mat(2 * normalized_L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')
