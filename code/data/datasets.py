import numpy as np
import sklearn.datasets
import csv


#### functions for reading from loooong file as stream

def getstuff(filename, rowlim):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in datareader:
            if count < rowlim:
                yield row
                count += 1
            else:
                return

def getdata(filename, rowlim):
    for row in getstuff(filename, rowlim):
        yield row

#####

def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    for i in range(X.shape[0]):
	X[i,:] -= mu
	for j in range(X.shape[1]):
	    X[i,j] /= std[j]
    return X

def load_data1():
    dataset = np.genfromtxt(open('data/ex2data1.txt','r'), delimiter=',', dtype='f8')
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
    dataset = np.genfromtxt(open('data/ex2data2.txt','r'), delimiter=',', dtype='f8')
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

def load_higgs(rowlim=10000):
    file_name = '../../datasets/HIGGS.csv'
    X, y = [], []
    #TODO: Fehler in dieser Zeile: X[0] wird ein array von STRINGs anstatt floats!    
    #for row in getdata(file_name, rowlim
    #    X.append(np.array([1.0] + row[1:]))
    raise NotImplementedError
    #    y.append(row[0])
    print type(X[0])
    print len(X[0])
    print X[0][0]
    print type(X[0][0])
    return X, y