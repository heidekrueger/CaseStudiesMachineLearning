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
w0 = np.array(XX[0].flat)

# store scores
scores = []

from SQN.SGD import SQN
from SQN.LogisticRegression import LogisticRegression
options = { 'dim':len(X[0]), 
                    'L': 10, 
                    'M': 5, 
                    'beta':5., 
                    'max_iter': 1000, 
                    'batch_size': 1000, 
                    'batch_size_H': 1000, 
                    'updates_per_batch': 1, 
                    'testinterval': 100, 
                    'w1': w0
                }

# cross-validation loop
for train, test in sss:
     
    sqn = SQN(options)
    logreg = LogisticRegression(lam_1 = 0.0, lam_2 = 0.0)
    
    x_train = [ np.array(XX[i].flat) for i in train ]
    y_train = [ y[i] for i in train ]

    x_test = [ np.array(XX[i].flat) for i in test ]
    y_test = [ y[i] for i in test ]
    logreg.w = sqn.solve(logreg.F, logreg.G, x_train, y_train)
    
    y_pred = logreg.predict(x_test)

    auc = roc_auc_score(y_test, y_pred)
    scores.append(auc)

# print scores
scores = np.asmatrix(scores)
print scores
