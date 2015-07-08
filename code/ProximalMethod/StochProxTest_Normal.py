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
    Ab = np.dot(A.T, b)
    A_sq = np.dot(A.T, A)
    x0 = np.ones((n, 1))
    L = linalg.eigvals(A_sq).real.max()
    
    return A, b, x0, l, L, batch_size, A_sq, Ab
    
def prox_comparison():
    
    import matplotlib.pyplot as plt
    import StochProxGrad as spg
    import StochProxMeth as spm
    import ProxMeth as pm
    import ProxGrad as pg
    from matplotlib2tikz import save as tikz_save
    
    fval_pm = pm.compute_0sr1(fn, gfn, x0, l_reg = l, tau = 1 / L, batch_size = batch_size)
    fval_pg = pg.proximal_gradient(fn, gfn, x0, 1 / L, l_reg = l, batch_size = batch_size)
    fval_spm = spm.compute_0sr1(f, gf, x0, X, y, l_reg = l, tau = 1 / L, batch_size = batch_size)
    fval_spg = spg.proximal_gradient(f, gf, x0, 1 / L, X, y, l_reg = l, batch_size = batch_size)
    fval_pm.insert(0, f(x0, X, y))
    fval_pg.insert(0, f(x0, X, y))
    fval_spm.insert(0, f(x0, X, y))
    fval_spg.insert(0, f(x0, X, y))
    
    line1, = plt.plot(range(len(fval_pm)), fval_pm, 'r', label = '0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_pg)), fval_pg, 'b', label = 'PG', lw = 2)
    line3, = plt.plot(range(len(fval_spm)), fval_spm, 'g', label = 'S0SR1', lw = 2)
    line4, = plt.plot(range(len(fval_spg)), fval_spg, 'k', label = 'SPG', lw = 2)
    
    
    #plt.xlim([0, 55])
    plt.yscale('log')
    #plt.ylim([1e1, 1e13])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3, line4])
    tikz_save( 'StochProx_150.tikz' );

    return
    
def f(x, X, y):
    
    temp = np.dot(X, x) - y
    
    return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
def gf(x, X, y):
    
    X_sq = np.dot(X.T, X)
    
    return  np.dot(X_sq, x) - np.dot(X.T, y)

if __name__ == "__main__":
    
    import numpy as np
    
    X, y, x0, l, L, batch_size, A_sq, Ab = initialize_lasso((1500, 3000), 0.1, 10)
    A = X.copy()
    b = y.copy()
    x0 *= 0.1
    
    def fn(x):
        temp = np.dot(A, x) - b
        return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
    def gfn(x):
        
        return  np.dot(A_sq, x) - Ab
    batch_size = 150
    prox_comparison()