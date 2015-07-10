# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:12:02 2015

@author: Fin Bauer
"""

from __future__ import division
from collections import deque


def initialize_lasso(size_A, l, max_iter, batch_size):
    import scipy.linalg as linalg
    
    m, n = size_A
    np.random.seed(0)
    A = np.random.normal(size = (m, n))
    b = np.random.normal(size = (m, 1))
    b_l_bfgs_b = b.copy()
    b_l_bfgs_b = b_l_bfgs_b.reshape(m)
    A_sq = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    x0 = 0.1 * np.ones((n, 1))
    bounds = 2 * n * [(0, None)]
    L = linalg.eigvals(A_sq).real.max()
    
    return (A, b, A_sq, Ab, x0, bounds, l, L, max_iter, batch_size)
    
def prox_comparison():
    
    import StochProxMeth as spm
    import StochProxGrad as spg
    
    _, fval_0sr1, xval_0sr1 = spm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L, 
                                           epsilon = 1e-6, max_iter = max_iter)
    _, fval_prox_grad, xval_prox_grad = spg.proximal_gradient(f, gf, x0, 1 / L, 
                                                          l_reg = l, 
                                                          max_iter = max_iter,
                                                          epsilon = 1e-6)

    return fval_0sr1, xval_0sr1, fval_prox_grad, xval_prox_grad
    

def f_conv_plot(fval_0sr1, fval_prox_grad):
    """
    
    """
    
    import matplotlib.pyplot as plt
    from matplotlib2tikz import save as tikz_save
    
    fval_0sr1.insert(0, f(x0))
    fval_prox_grad.insert(0, f(x0))
    
    line1, = plt.plot(range(len(fval_0sr1)), fval_0sr1, 'r', label = '0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_prox_grad)), fval_prox_grad, 'b', label = 'ProxGrad', lw = 2)
    line3, = plt.plot(range(len(fval_l_bfgs_b)), fval_l_bfgs_b, 'g', label = 'L-BFGS-B', lw = 2)
    plt.xscale('log')
    #plt.xlim([0, 1e2])
    plt.yscale('log')
    #plt.ylim([0, 1e5])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    #tikz_save( 'myfile2.tikz' );
    plt.show()
    
    return
    
    
def x_conv_plot(xval_0sr1, xval_prox_grad):
    """
    """
    
    import matplotlib.pyplot as plt
    from matplotlib2tikz import save as tikz_save
    
    d_0sr1, d_prox_grad, d_l_bfgs_b = [], [], []
    
    for j in range(len(xval_0sr1)):
        d_0sr1.append(np.linalg.norm(xval_0sr1[j] - xval_0sr1[-1]))
    
    for j in range(len(xval_prox_grad)):
        d_prox_grad.append(np.linalg.norm(xval_prox_grad[j]-xval_prox_grad[-1]))
        
    for j in range(len(xval_l_bfgs_b)):
        d_l_bfgs_b.append(np.linalg.norm(xval_l_bfgs_b[j]-xval_l_bfgs_b[-1]))
    
    q_0sr1, q_prox_grad, q_l_bfgs_b = [], [], []
    
    for i in range(len(d_0sr1) - 1):
        q_0sr1.append(d_0sr1[i+1] / d_0sr1[i])
    
    for i in range(len(d_prox_grad) - 1):
        q_prox_grad.append(d_prox_grad[i+1] / d_prox_grad[i])
        
    for i in range(len(d_l_bfgs_b) - 1):
        q_l_bfgs_b.append(d_l_bfgs_b[i+1] / d_l_bfgs_b[i])
    
    line1, = plt.plot(range(len(q_0sr1)), q_0sr1, 'r', label = '0SR1', lw = 2)
    line2, = plt.plot(range(len(q_prox_grad)), q_prox_grad, 'b', label = 'ProxGrad', lw = 2)
    line3, = plt.plot(range(len(q_l_bfgs_b)), q_l_bfgs_b, 'g', label = 'L-BFGS-B', lw = 2)
    
    #plt.xscale('log')
    plt.xlim([0, 600])
    #plt.yscale('log')
    #plt.ylim([0, 1e5])
    plt.ylabel('Convergence Factor')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    tikz_save( 'conv.tikz' );
    
    return
    
    
def conv_time():
    
    import StochProxMeth as spm
    import StochProxGrad as spg
    import time
    
    t_0sr1 = 0
    t_prox_grad = 0
    t_l_bfgs_b = 0
    
    for i in range(1):
    
        t0 = time.time()
        _, _, iter_0sr1 = spm.compute_0sr1(f, gf, x0, X, y, l_reg = l, tau = 1 / L, 
                                             timing = 1, epsilon = 1e-6, 
                                             max_iter = max_iter, batch_size = batch_size)
        t_0sr1 += time.time() - t0
        print(t_0sr1)
        print(iter_0sr1)
        t0 = time.time()
        _, _, iter_prox_grad = spg.proximal_gradient(f, gf, x0, 1 / L, X, y,
                                                       l_reg = l, timing = 1, 
                                                       max_iter = max_iter, 
                                                       epsilon = 1e-6, batch_size = batch_size)
        t_prox_grad += time.time() - t0
        print(t_prox_grad)
        print(iter_prox_grad)
        
    
    t_0sr1 /= 5
    t_prox_grad /= 5
    t_l_bfgs_b /= 5
    
    return t_0sr1, t_prox_grad, iter_0sr1, iter_prox_grad

def f(x, X, y):
    
    temp = np.dot(X, x) - y
    
    return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
def gf(x, X, y):
    
    X_sq = np.dot(X.T, X)
    
    return  np.dot(X_sq, x) - np.dot(X.T, y)

if __name__ == "__main__":
    
    import numpy as np
    
    X, y, A_sq, Ab, x0, bounds, l, L, max_iter, batch_size = initialize_lasso((1500, 3000),
                                                                0.1, 1e7, 150)
    A = X.copy()
    b = y.copy()
    
    def fn(x):
        
        temp = np.dot(A, x) - b
        
        return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
    def gfn(x):
        
        return  np.dot(A_sq, x) - Ab
    
    #fval_0sr1, xval_0sr1, fval_prox_grad, xval_prox_grad = prox_comparison()
    #f_conv_plot(fval_0sr1, fval_prox_grad)
    #x_conv_plot(xval_0sr1, xval_prox_grad)
    t_0sr1, t_prox_grad, iter_0sr1, iter_prox_grad = conv_time()
    