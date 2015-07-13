# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:12:02 2015

@author: Fin Bauer
"""

from __future__ import division


def initialize_lasso(size_A, l, max_iter, batch_size):
    import scipy.linalg as linalg

    m, n = size_A
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m, 1))
    x0 = 0.1 * np.ones((n, 1))
    A_sq = np.dot(A.T, A)
    L = linalg.eigvals(A_sq).real.max()

    return (A, b, x0, l, L, max_iter, batch_size)


def conv_time():

    import StochProxMeth as spm
    import StochProxGrad as spg
    import time

    t_0sr1 = 0
    t_prox_grad = 0
    t_l_bfgs_b = 0

    for i in range(5):

        t0 = time.time()
        _, _, iter_0sr1 = spm.compute_0sr1(f, gf, x0, X, y, l_reg=l,
                                           tau=1 / L, timing=1,
                                           epsilon=1e-6, max_iter=max_iter,
                                           batch_size=batch_size)
        t_0sr1 += time.time() - t0
        print(t_0sr1)
        print(iter_0sr1)
        t0 = time.time()
        _, _, iter_prox_grad = spg.proximal_gradient(f, gf, x0, 1 / L, X, y,
                                                     l_reg=l, timing=1,
                                                     max_iter=max_iter,
                                                     epsilon=1e-6,
                                                     batch_size=batch_size)
        t_prox_grad += time.time() - t0
        print(t_prox_grad)
        print(iter_prox_grad)


    t_0sr1 /= 5
    t_prox_grad /= 5
    t_l_bfgs_b /= 5

    return t_0sr1, t_prox_grad, iter_0sr1, iter_prox_grad


def f(x, X, y):

    temp = np.dot(X, x) - y

    return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord=1)


def gf(x, X, y):

    X_sq = np.dot(X.T, X)

    return  np.dot(X_sq, x) - np.dot(X.T, y)


if __name__ == "__main__":

    import numpy as np

    X, y, x0, l, L, max_iter, batch_size = initialize_lasso((1500, 3000),
                                                            0.1, 1e7, 150)

    t_0sr1, t_prox_grad, iter_0sr1, iter_prox_grad = conv_time()
