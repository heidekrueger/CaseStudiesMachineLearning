# -*- coding: utf-8 -*-
"""
Created on Tuesday 9 July

@author: Stanislas Chambon
"""

from ProxEegTest import prox_meth_lr, predict
import numpy as np
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from time import time


# load datasets
X = np.load('../datasets/eeg_data.npy')
y = np.load('../datasets/eeg_label.npy')

# standardizing data
nn = np.mean(X, axis=1)
X -= nn[:, None]

nn = np.sqrt(np.sqrt(np.sum(X * X, axis=1)))
X /= nn[:, None]

# features extraction
XX = np.concatenate((np.std(X, axis=1)[:, None],
                     stats.kurtosis(X, axis=1)[:, None]), axis=1)

# number of runs for cv
n_cv = 5

# creation of a cross-validation object
sss = StratifiedShuffleSplit(y, n_cv, train_size=0.8, random_state=0)

# parameter to modify
w0 = X[0, :]
l_reg = 0.1
tau = 0.001
batch_size = 100

# store scores
scores = []

# cross-validation loop
t0 = time()
l_nzc = []

c_cv = 0
for train, test in sss:
    c_cv += 1
    print "iter :", c_cv

    x_train = X[train, :]
    y_train = y[train]

    x_test = X[test, :]
    y_test = y[test]

    w = prox_meth_lr(x_train, y_train, w0,
                     l_reg=l_reg, tau=tau, batch_size=batch_size)

    # compute the sparsity of w
    nzc = 0

    for i in range(0, len(w)):
        if w[i] <= 1.e-07:
            nzc += 1
    l_nzc.append(nzc)

    # add a 1 - pred : to predict the class of samples followed by s. o.
    y_pred = 1 - predict(w, x_test)

    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)

# compute running time
dt = (time() - t0) / float(n_cv)

# transform scores / l_nzc
scores = np.asmatrix(scores)
l_nzc = np.asmatrix(l_nzc)

print "mean score    :", scores.mean()
print "std, score    :", scores.std()
print "running time  :", dt
print "mean sparsity :", l_nzc.mean() / float(len(w))
