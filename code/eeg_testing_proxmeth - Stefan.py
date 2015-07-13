# -*- coding: utf-8 -*-
"""
Created on Tuesday 9 July

@author: Stanislas Chambon, Stefan Heidekrueger
"""

from ProxEegTest import prox_meth_lr, predict
import numpy as np
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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
# XX = np.concatenate((np.std(X, axis=1)[:, None],
#                      stats.kurtosis(X, axis=1)[:, None]), axis=1)
XX = SelectKBest(f_classif,k=2).fit_transform(X,y)
# XX = X[:, 500:600]
print XX.shape
# number of runs for cv
n_cv = 1

# creation of a cross-validation object
sss = StratifiedShuffleSplit(y, n_cv, train_size=0.8, random_state=0)

# parameter to modify
w0 = XX[0, :]
l_reg = 0.05
tau = 0.001
batch_size = 1000

# store scores
scores = []
accs = []

# cross-validation loop
l_nzc = []

# time counting
t0 = time()

c_cv = 0
for train, test in sss:
    c_cv += 1
    print "cv iter :", c_cv

    x_train = XX[train, :]
    y_train = y[train]

    x_test = XX[test, :]
    y_test = y[test]

    w = prox_meth_lr(x_train, y_train, w0,
                     l_reg=l_reg, tau=tau, batch_size=batch_size,
                     max_iter=100)

    # compute the sparsity of w
    nzc = 0

    for i in range(0, len(w)):
        if w[i] <= 1.e-07:
            nzc += 1
    l_nzc.append(nzc)

    y_pred = predict(w, x_test)
    y_pred_bin = [int(p+.5) for p in y_pred]
    acc = accuracy_score(y_test,y_pred_bin)
    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)
    accs.append(acc)

# compute running time
dt = (time() - t0) / float(n_cv)

# transform scores / l_nzc
scores = np.asmatrix(scores)
accs = np.asmatrix(accs)
l_nzc = np.asmatrix(l_nzc)

print "mean score    :", scores.mean()
print "std, score    :", scores.std()
print "mean accur    :", accs.mean()
print "running time  :", dt
print "mean sparsity :", l_nzc.mean() / float(len(w))
