# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:13:15 2015

@author: Fin Bauer
"""
# from future import division


def prox_meth_lr(X, y, x0, l_reg=1., tau=0.01, batch_size=1, max_iter=1000):
    """
    Stochast Proximal Method (0SR1)

    Input:
    X = training set (2-D np array)
    y = labels (2-D np array)
    x0 = starting point (2-D np array)

    Options:
    l_reg = lambda
    tau = preferably 1 / L, with L Lipschitz constant of gradient of log reg
    batch_size = batch size

    Return:
    w_opt = Optimal parameter
    """
    import ProximalMethod.Prox_LogReg as lr
    import ProximalMethod.StochProxMeth as spm

    _, w_opt,_ = spm.compute_0sr1(lr.F, lr.G, x0, X, y, l_reg=l_reg,
                                tau=tau, batch_size=batch_size, max_iter = max_iter)

    return w_opt


def prox_grad_lr(X, y, x0, l_reg=1., tau=0.01, batch_size=1, max_iter = 1000):
    """
    Stochastic Proximal Gradient

    Input:
    X = training set (2-D np array)
    y = labels (2-D np array)
    x0 = starting point (2-D np array)

    Options:
    l_reg = lambda
    tau = preferably 1 / L, with L Lipschitz constant of gradient of log reg
    batch_size = batch size

    Return:
    w_opt = Optimal parameter
    """

    import Prox_LogReg as lr
    import StochProxGrad as spg

    _, w_opt,_ = spg.compute_0sr1(lr.F, lr.G, x0, tau, X, y, l_reg=l_reg,
                                batch_size=batch_size, max_iter=max_iter)

    return w_opt


def predict(w, X):
    """
    Prediction for logistic regression

    Input:
    w = trained parameter (2-D np array)
    X = test data (2-D np array)

    Return:
    pred = predicted labels
    """
    import numpy as np
    # import Prox_LogReg as lr
    import ProximalMethod.Prox_LogReg as lr

    m, n = np.shape(X)
    pred = np.zeros(m)

    for i in range(m):
        pred[i] = lr.sigmoid(w, X[i].reshape(n, 1))

    return pred
