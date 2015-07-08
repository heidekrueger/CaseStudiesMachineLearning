# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:27:00 2015

@author: Fin Bauer
"""

from __future__ import division


def read_fval(x):
    
    f_current = f_l_bfgs_b(x)
    fval_l_bfgs_b.append(f_current)
    return

def initialize_lasso(size_A, l):
    import scipy.linalg as linalg
    
    m, n = size_A
    np.random.seed(0)
    A = np.random.normal(size = (m, n))
    b = np.random.normal(size = (m, 1))
    b_l_bfgs_b = b.copy()
    b_l_bfgs_b = b_l_bfgs_b.reshape(m)
    A_sq = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    Ab_l_bfgs_b = np.concatenate((np.dot(A.T, b_l_bfgs_b),-np.dot(A.T,b_l_bfgs_b))).T
    x0 = np.ones((n, 1))
    x0_l_bfgs_b = np.concatenate((np.ones(n), np.zeros(n)))
    bounds = 2 * n * [(0, None)]
    L = linalg.eigvals(A_sq).real.max()
    
    return A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, l, L
    
def prox_comparison():
    
    import matplotlib.pyplot as plt
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    from matplotlib2tikz import save as tikz_save
    
    fval_0sr1 = pm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L)
    fval_prox_grad = pg.proximal_gradient(f, gf, x0, 1 / L, l_reg = l)
    spopt.fmin_l_bfgs_b(f_l_bfgs_b, x0_l_bfgs_b, gf_l_bfgs_b, 
                        bounds = bounds, callback = read_fval, maxiter = 60)
    fval_0sr1.insert(0, f(x0))
    fval_prox_grad.insert(0, f(x0))
    fval_l_bfgs_b.insert(0, f_l_bfgs_b(x0_l_bfgs_b))
    line1, = plt.plot(range(len(fval_0sr1)), fval_0sr1, 'r', label = '0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_prox_grad)), fval_prox_grad, 'b', label = 'ProxGrad', lw = 2)
    line3, = plt.plot(range(len(fval_l_bfgs_b)), fval_l_bfgs_b, 'g', label = 'L-BFGS-B', lw = 2)
    plt.xlim([0, 60])
    plt.yscale('log')
    plt.ylim([0, 1e5])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    tikz_save( 'myfile2.tikz' );

    return

if __name__ == "__main__":
    
    import numpy as np
    
    A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, l, L = initialize_lasso((1500, 3000), 0.1)
    fval_l_bfgs_b = []
    x0 *= 0.1
    x0_l_bfgs_b *= 0.1
    
    def f(x):
        temp = np.dot(A, x) - b
        return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
    def gf(x):
        
        return  np.dot(A_sq, x) - Ab
        
    def f_l_bfgs_b(x):
        n = len(x)
        temp = np.dot(A, x[:np.floor(n/2)]) - np.dot(A, x[np.floor(n/2):]) - b_l_bfgs_b
        return 1/2 * np.dot(temp.T, temp) + l * sum(abs(x))
        
    def gf_l_bfgs_b(x):
        n = len(x)
        temp = np.dot(A_sq, x[:np.floor(n/2)]) - np.dot(A_sq, x[np.floor(n/2):])
        return np.concatenate((temp, -temp)).T - Ab_l_bfgs_b + l
    
    prox_comparison()