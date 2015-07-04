# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:27:00 2015

@author: Fin Bauer
"""

import numpy as np
import scipy as sp

def test_l_bfgs_b(x):
    f_current = f_l_bfgs_b(x)
    f_val_l_bfgs_b.append(f_current)
    return
    
def test_Lasso(A, b, f):
    from sklearn.linear_model import Lasso
    lars = Lasso()
    lars.fit(A, b)
    n_iter = int(np.floor(lars.n_iter_ / 10))
    f_val = np.zeros((n_iter))
    for i in range(n_iter):
        lars = Lasso(max_iter = (i+1) * 10)
        lars.fit(A, b)
        f_val[i] = f(lars.coef_)
    return f_val



if __name__ == "__main__":
    
    A = np.random.normal(size = (150, 300))
    b = np.random.normal(size=(150))
    A_sq = np.dot(A.T, A)
    Ab = np.concatenate((np.dot(A.T, b),-np.dot(A.T,b))).T
    
    def f(x):
        temp = np.dot(A, x) - b
        return 1 / 2 * np.dot(temp, temp) + np.linalg.norm(x, ord = 1)
        
    def f_l_bfgs_b(x):
        n = len(x)
        temp = np.dot(A, x[:np.floor(n/2)]) - np.dot(A, x[np.floor(n/2):]) - b
        return 1/2 * np.dot(temp.T, temp) + sum(abs(x))
        
    def gf_l_bfgs_b(x):
        n = len(x)
        x=x
        temp = np.dot(A_sq, x[:np.floor(n/2)]) - np.dot(A_sq, x[np.floor(n/2):])
        return np.concatenate((temp, -temp)).T - Ab + np.ones(n)
    
    x0 = np.ones(600)
    bounds = [(0, None)] * 600
    
    f_val_lasso = test_Lasso(np.sqrt(150) * A, np.sqrt(150) * b, f)
    f_val_l_bfgs_b = []
    x_bfgs, f, d = sp.optimize.fmin_l_bfgs_b(f_l_bfgs_b,x0,gf_l_bfgs_b,bounds = bounds, callback = test_l_bfgs_b)