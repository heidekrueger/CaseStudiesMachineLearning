import numpy as np
import sklearn as sk


def load_data1():
    dataset = np.genfromtxt(open('ex2data1.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2]
    return X, z

def load_data2():
	dataset = np.genfromtxt(open('ex2data2.txt','r'), delimiter=',', dtype='f8')
    X = dataset[:,:2]
    z = dataset[:,2]
    return X, z

