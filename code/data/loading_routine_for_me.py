"""
small routine for me
"""

import numpy as np

data_name = 'data_train.npy'
label_name = 'label_train.npy'

X = np.load(data_name)
y = np.load(label_name)

print X.shape
print y.shape

perm = np.random.randint(0, len(y), len(y))

X = X[perm, :]
y = y[perm]

print X.shape
print y.shape

np.save('data', X)
np.save('labels', y)
