# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:43:42 2015

@author: Fin Bauer
"""


def initialize_lasso(size_A, l, batch_size):
    import scipy.linalg as linalg
    
    m, n = size_A
    np.random.seed(0)
    A = np.random.normal(size = (m, n))
    b = np.random.normal(size = (m, 1))
    A_sq = np.dot(A.T, A)
    x0 = np.ones((n, 1))
    L = linalg.eigvals(A_sq).real.max()
    
    return A, b, x0, l, L, batch_size
    
def prox_comparison():
    
    import matplotlib.pyplot as plt
    import StochProxGrad as spg
    import StochProxMeth as spm
    from matplotlib2tikz import save as tikz_save
    
    fval_spm = spm.compute_0sr1(f, gf, x0, X, y, l_reg = l, tau = 1 / L, batch_size = batch_size)
    fval_spg = spg.proximal_gradient(f, gf, x0, 1 / L, X, y, l_reg = l, batch_size = batch_size)
    fval_spm.insert(0, f(x0, X, y))
    fval_spg.insert(0, f(x0, X, y))
    
    line1, = plt.plot(range(len(fval_spm)), fval_spm, 'r', label = 'S0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_spg)), fval_spg, 'b', label = 'SPG', lw = 2)
    
    #plt.xlim([0, 55])
    plt.yscale('log')
    #plt.ylim([1e1, 1e13])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2])
    #tikz_save( 'myfile2.tikz' );

    return
    
def f(x, X, y):
    
    temp = np.dot(X, x) - y
    
    return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
def gf(x, X, y):
    
    X_sq = np.dot(X.T, X)
    
    return  np.dot(X_sq, x) - np.dot(X.T, y)

if __name__ == "__main__":
    
    import numpy as np
    
    X, y, x0, l, L, batch_size = initialize_lasso((1500, 3000), 0.1, 10)
    x0 *= 0.1
    
    prox_comparison()