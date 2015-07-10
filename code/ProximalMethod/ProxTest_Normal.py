# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:27:00 2015

@author: Fin Bauer
"""

from __future__ import division
from collections import deque


def read_fval(x):
    
    f_current = f_l_bfgs_b(x)
    fval_l_bfgs_b.append(f_current)
    xval = x.copy()
    xval_l_bfgs_b.append(xval)
    
    return

def initialize_lasso(size_A, l, max_iter):
    import scipy.linalg as linalg
    
    m, n = size_A
    np.random.seed(0)
    A = np.random.normal(size = (m, n))
    b = np.random.normal(size = (m, 1))
    b_l_bfgs_b = b.copy()
    b_l_bfgs_b = b_l_bfgs_b.reshape(m)
    A_sq = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    Ab_l_bfgs_b = np.concatenate((np.dot(A.T, b_l_bfgs_b),
                                  -np.dot(A.T,b_l_bfgs_b))).T
    x0 = 0.1 * np.ones((n, 1))
    x0_l_bfgs_b = 0.1 * np.concatenate((np.ones(n), np.zeros(n)))
    bounds = 2 * n * [(0, None)]
    L = linalg.eigvals(A_sq).real.max()
    fval_l_bfgs_b = []
    xval_l_bfgs_b = deque([x0_l_bfgs_b], 500)
    
    return (A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, 
            l, L, fval_l_bfgs_b, xval_l_bfgs_b, max_iter)
    
def prox_comparison():
    
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    
    _, fval_0sr1, xval_0sr1 = pm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L, 
                                           epsilon = 1e-6, max_iter = max_iter)
    _, fval_prox_grad, xval_prox_grad = pg.proximal_gradient(f, gf, x0, 1 / L, 
                                                          l_reg = l, 
                                                          max_iter = max_iter,
                                                          epsilon = 1e-6)
    spopt.fmin_l_bfgs_b(f_l_bfgs_b, x0_l_bfgs_b, gf_l_bfgs_b, 
                        bounds = bounds, callback = read_fval, 
                        maxiter = max_iter)

    return fval_0sr1, xval_0sr1, fval_prox_grad, xval_prox_grad
    

def f_conv_plot(fval_0sr1, fval_prox_grad):
    """
    
    """
    
    import matplotlib.pyplot as plt
    from matplotlib2tikz import save as tikz_save
    
    fval_0sr1.insert(0, f(x0))
    fval_prox_grad.insert(0, f(x0))
    fval_l_bfgs_b.insert(0, f_l_bfgs_b(x0_l_bfgs_b))
    fval_l_bfgs_b.insert(0, f_l_bfgs_b(x0_l_bfgs_b))
    
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
    
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    import time
    
    t_0sr1 = 0
    t_prox_grad = 0
    t_l_bfgs_b = 0
    
    for i in range(2):
    
        t0 = time.time()
        _, _, _, iter_0sr1 = pm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L, 
                                             timing = 1, epsilon = 1e-6, 
                                             max_iter = max_iter)
        t_0sr1 += time.time() - t0
        
        t0 = time.time()
        _, _, _, iter_prox_grad = pg.proximal_gradient(f, gf, x0, 1 / L, 
                                                       l_reg = l, timing = 1, 
                                                       max_iter = max_iter, 
                                                       epsilon = 1e-6)
        t_prox_grad += time.time() - t0
        
        t0 = time.time()
        spopt.fmin_l_bfgs_b(f_l_bfgs_b, x0_l_bfgs_b, gf_l_bfgs_b, 
                            bounds = bounds, maxiter = max_iter)
        t_l_bfgs_b += time.time() - t0
    
    t_0sr1 /= 10
    t_prox_grad /= 10
    t_l_bfgs_b /= 10
    
    return t_0sr1, t_prox_grad, t_l_bfgs_b, iter_0sr1, iter_prox_grad


if __name__ == "__main__":
    
    import numpy as np
    
    (A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, l, L, 
     fval_l_bfgs_b, xval_l_bfgs_b, max_iter) = initialize_lasso((15, 30),
                                                                0.1, 1e6)
    
    def f(x):
        
        temp = np.dot(A, x) - b
        
        return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
    def gf(x):
        
        return  np.dot(A_sq, x) - Ab
        
    def f_l_bfgs_b(x):
        
        n = len(x)
        temp = (np.dot(A, x[:np.floor(n/2)]) - np.dot(A, x[np.floor(n/2):])- 
                b_l_bfgs_b)
                
        return 1/2 * np.dot(temp.T, temp) + l * sum(abs(x))
        
    def gf_l_bfgs_b(x):
        
        n = len(x)
        temp = (np.dot(A_sq, x[:np.floor(n/2)]) - 
                np.dot(A_sq, x[np.floor(n/2):]))
                      
        return np.concatenate((temp, -temp)).T - Ab_l_bfgs_b + l
    
    #fval_0sr1, xval_0sr1, fval_prox_grad, xval_prox_grad = prox_comparison()
    #f_conv_plot(fval_0sr1, fval_prox_grad)
    #x_conv_plot(xval_0sr1, xval_prox_grad)
    t_0sr1, t_prox_grad, t_l_bfgs_b, iter_0sr1, iter_prox_grad = conv_time()
    