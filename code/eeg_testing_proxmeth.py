# -*- coding: utf-8 -*-
"""
Created on Tuesday 9 July

@author: Stanislas Chambon
"""

from ProxEegTest import prox_meth_lr, predict
import numpy as np
from scipy import stats

# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

# from scipy import stats

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

# creation of a cross-validation object
sss = StratifiedShuffleSplit(y, 5, train_size=0.8, random_state=0)

# parameter to modify
w0 = XX[0, :]
l_reg = .01
tau = 0.001
batch_size = 1000

# store scores
scores = []

# cross-validation loop
for train, test in sss:
    x_train = XX[train, :]
    y_train = y[train]

    x_test = XX[test, :]
    y_test = y[test]

    w = prox_meth_lr(x_train, y_train, w0,
                     l_reg=l_reg, tau=tau, batch_size=batch_size)
    y_pred = predict(w, x_test)

    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)

# print scores
scores = np.asmatrix(scores)
print scores
