# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:16:11 2015

@author: Fin Bauer
"""

import numpy as np

############################## Objective ######################################

def F(w, X, y):
    """
    Overall objective function
    """
    n, m = np.shape(X)
    
    return sum([f(w, X[i].reshape(m, 1), float(y[i])) for i in range(n)]) / n

    
def f(w, X, y):
    """
    Loss functions as column vector
    """
    
    hyp = sigmoid(w, X)
    
    return -1*float(y * np.log(hyp) + (1 - y) * (np.log(1 - hyp))) # *(-1)???


def sigmoid(w, X):
    """
    sigmoid function
    """
    
    z = np.dot(X.T, w)
    z = np.sign(z) * min([np.abs(z), 30])
    
    return float(1 / (1.0 + np.exp(-z)))


############################# Gradient ########################################

def G(w, X, y):
    """
    Gradient of overall objective Function
    """
    n, m = np.shape(X)
    
    return sum([g(w, X[i].reshape(m, 1), float(y[i])) for i in range(n)]) / n


def g(w, X, y):
    """
    Gradient of loss function f
    """
    
    hyp = sigmoid(w, X)
    
    return (hyp - y) * X