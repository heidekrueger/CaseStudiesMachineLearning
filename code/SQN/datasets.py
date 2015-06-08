import numpy as np
import sklearn as sk

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
	x = np.array(list(X[i,:]))
	x.shape = (len(x), 1)
	X_new.append(x)
    return X_new, list(z)

def load_data2():
    dataset = np.genfromtxt(open('ex2data2.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2]
    X = np.concatenate( (np.ones( (len(z), 1)), X), axis=1)
    return X, z

