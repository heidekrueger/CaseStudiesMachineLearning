from SQN.LogisticRegression import LogisticRegression
from SQN.SQN import SQN
import numpy as np
import data.datasets as datasets
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy import stats
from sklearn.metrics import roc_auc_score

print "Loading eeg data..."
X, y = datasets.load_eeg()
print "eeg data loaded."

print "Data dim : ", X.shape
print "Label dim : ", y.shape

# standardizing data
nn = np.mean(X, axis=1)
X -= nn[:, None]

nn = np.sqrt(np.sqrt(np.sum(X * X, axis=1)))
X /= nn[:, None]

# features extraction
XX = np.concatenate((np.std(X, axis=1)[:, None],
                     stats.kurtosis(X, axis=1)[:, None]), axis=1)

# sklearn test

# create cross-valalidation sets
sss = StratifiedShuffleSplit(y, 5, train_size=0.8, random_state=0)

# creation of logreg object
lr = LR()

# scores
scores = cross_val_score(lr, XX, y,
                         cv=sss, scoring='roc_auc', n_jobs=5)

print scores

# data transformation
data = []
label = []

for i in range(0, len(y)):
    data.append(XX[i, :])
    label.append(y[i])

print len(data)
print len(label)

# our test
print "our test begins"
logreg = LogisticRegression(lam_1=0.0, lam_2=0.0)
options = {'dim': len(data[0]),
           'L': 10,
           'max_iter': 100,
           'batch_size': 100,
           'batch_size_H': 20,
           'beta': 10,
           'M': 3}
sqn = SQN(options)

print "SQN starting"
sqn.solve(logreg.F, logreg.g, data, label)
print "SQN finished"

print 'predicting'
w = sqn.get_position()
logreg.w = w
label_pred = logreg.predict(data)
print 'predicted'

auc = roc_auc_score(label, label_pred)
print auc
