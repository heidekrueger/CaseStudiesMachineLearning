import numpy as np
import sklearn.datasets

def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    for i in range(X.shape[0]):
	X[i,:] -= mu
	for j in range(X.shape[1]):
	    X[i,j] /= std[j]
    return X

def load_data1():
    dataset = np.genfromtxt(open('ex2data1.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2][:,np.newaxis]
    
    X = normalize(X)
    
    X = np.concatenate( (np.ones( (len(z), 1)), X), axis=1)
    X_new = []
    for i in range(len(z)):
	x = np.array(list(X[i,:].flatten()))
	X_new.append(x)
    return X_new, list(z)

def load_data2():
    dataset = np.genfromtxt(open('ex2data2.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2][:,np.newaxis]
    
    X = normalize(X)
    
    X = np.concatenate( (np.ones( (len(z), 1)), X), axis=1)
    X_new = []
    for i in range(len(z)):
	x = np.array(list(X[i,:].flatten()))
	X_new.append(x)
    return X_new, list(z)

def load_iris():
	iris = sklearn.datasets.load_iris()
	X, y = [], []
	for i in range(len(iris.target)):
		if iris.target[i] != 2:
			X.append(np.array([1] + list(iris.data[i])))
			y.append(iris.target[i])
	return X, y