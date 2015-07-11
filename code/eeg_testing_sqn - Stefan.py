# -*- coding: utf-8 -*-
"""
Created on Tuesday 9 July

@author: Roland Halbich / Stanislas Chambon
"""

import numpy as np
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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
                      stats.kurtosis(X, axis=1)[:, None], stats.skew(X, axis=1)[:, None]), axis=1)
#XX = SelectKBest(f_classif, k=2).fit_transform(X, y)

print XX.shape
# number of runs for cv
n_cv = 1

# creation of a cross-validation object
sss = StratifiedShuffleSplit(y, n_cv, train_size=0.8, random_state=0)

# parameter to modify
w0 = np.array(XX[0].flat)

# store scores
scores = []
accs = []

options = {'dim': len(X[0]),
           'L': 10,
           'M': 5,
           'beta': 5.,
           'max_iter': 1000,
           'batch_size': 100,
           'batch_size_H': 500,
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

    y_pred = [p for p in logreg.predict(x_test)]
    y_pred_bin = [1 if p>0 else 0 for p in y_pred]

    print y_test[:20], y_pred[:20], y_pred_bin[:20]


    acc = accuracy_score(y_test, y_pred_bin)
    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)
    accs.append(acc)

# compute running time
dt = (time() - t0) / float(n_cv)


# print scores
scores = np.asmatrix(scores)
accs = np.asmatrix(accs)

print "mean score    :", scores.mean()
print "std, score    :", scores.std()
print "mean acc      :", accs.mean()
print "std, acc    :", accs.std()
print "running time  :", dt
