# -*- coding: utf-8 -*-
"""
Created on Tuesday 9 July

@author: Roland Halbich / Stanislas Chambon
"""

import numpy as np
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from SQN.SGD import SQN
from SQN.LogisticRegression import LogisticRegression
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
w0 = np.array(XX[0].flat)

# store scores
scores = []

options = {'dim': len(X[0]),
           'L': 10,
           'M': 5,
           'beta': 5.,
           'max_iter': 1000,
           'batch_size': 1000,
           'batch_size_H': 1000,
           'updates_per_batch': 1,
           'testinterval': 100,
           'w1': w0}

# time counting
t0 = time()

# cross-validation loop
for train, test in sss:
    sqn = SQN(options)
    logreg = LogisticRegression(lam_1=0.0, lam_2=0.0)

    x_train = [np.array(XX[i].flat) for i in train]
    y_train = [y[i] for i in train]

    x_test = [np.array(XX[i].flat) for i in test]
    y_test = [y[i] for i in test]
    logreg.w = sqn.solve(logreg.F, logreg.G, x_train, y_train)

    y_pred = 1 - logreg.predict(x_test)

    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)

# compute running time
dt = (time() - t0) / float(n_cv)


# print scores
scores = np.asmatrix(scores)

print "mean score    :", scores.mean()
print "std, score    :", scores.std()
print "running time  :", dt
