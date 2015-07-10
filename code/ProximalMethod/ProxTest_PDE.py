# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:21:19 2015

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

def u(l, m, n, alpha, beta, gamma, sigma):
    h = 1 / (l + 1)
    Y = (m+1)*h
    Z = (n+1)*h
    x1 = np.linspace(h, 1, l, endpoint=False)
    y1 = np.linspace(h, (m+1)*h, m, endpoint=False)
    z1 = np.linspace(h, (n+1)*h, n, endpoint=False)
    x, y, z = np.meshgrid(x1, y1, z1)
    uval = x*(x-1)*y*(y-Y)*z*(z-Z)*np.exp(-1/2*sigma**2*((x-alpha)**2+(y-beta)**2+(z-gamma)**2))
    return uval


def initialize_lasso(size_A, lam, max_iter, alpha=0.4, beta=0.7, gamma=0.5, sigma=50):
    
    np.random.seed(1)
    l, m, n = size_A
    moT = (-1,) * l
    moW = (-1,) * m
    moA = (-1,) * n
    I = spa.eye(m = l, n = l, dtype = np.double)
    IW = spa.eye(m = m, n = m, dtype = np.double)
    T = spa.spdiags(((6,) * l, moT, moT), (0, -1, 1), l, l)
    CW = spa.spdiags((moW, moW), (-1, 1), m, m)
    W = spa.kron(IW, T) + spa.kron(CW, I)
    IWA = spa.eye(m = l * m, n = l * m, dtype = np.double)
    IA = spa.eye(m = n, n = n, dtype = np.double)
    CA = spa.spdiags((moA, moA), (-1, 1), n, n)
    A = spa.kron(IA, W) + spa.kron(CA, IWA)
    b = u(l, m, n, alpha, beta, gamma, sigma).reshape(l*m*n,1)
    b_l_bfgs_b = b.copy()
    b_l_bfgs_b = b_l_bfgs_b.reshape(len(b))
    A_sq = A.dot(A)
    Ab = A.dot(b).reshape(l * m * n, 1)
    Ab_l_bfgs_b = np.concatenate((A.dot(b_l_bfgs_b),-A.dot(b_l_bfgs_b)))
    x0 = abs(np.random.normal(size = (l * m * n, 1)))
    x0_l_bfgs_b = np.concatenate((x0.reshape(l*m*n), np.zeros(l * m * n)))
    bounds = 2 * l * m * n * [(0, None)]
    L, _ = splin.eigs(A_sq, k=1)
    L=float(abs(L))
    fval_l_bfgs_b = []
    xval_l_bfgs_b = deque([x0_l_bfgs_b], 500)
    
    return A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, lam, L, fval_l_bfgs_b, xval_l_bfgs_b, max_iter
    
def prox_comparison():
    
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    
    _, fval_0sr1, xval_0sr1, _ = pm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L, 
                                           epsilon = 1e-6, max_iter = max_iter)
    _, fval_prox_grad, xval_prox_grad, _ = pg.proximal_gradient(f, gf, x0, 1 / L, 
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
    #plt.xscale('log')
    #plt.xlim([0, 1e2])
    plt.yscale('log')
    #plt.ylim([0, 1e5])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    tikz_save( 'ProxPDE.tikz' );
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
    plt.xlim([0, 20])
    #plt.yscale('log')
    #plt.ylim([0, 1e5])
    plt.ylabel('Convergence Factor')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    tikz_save( 'convPDE.tikz' );
    
    return
    
    
def conv_time():
    
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    import time
    
    t_0sr1 = 0
    t_prox_grad = 0
    t_l_bfgs_b = 0
    
    for i in range(5):
    
        t0 = time.time()
        _, _, _, iter_0sr1 = pm.compute_0sr1(f, gf, x0, l_reg = l, tau = 1 / L, 
                                             timing = 1, epsilon = 1e-6, 
                                             max_iter = max_iter)
        t_0sr1 += time.time() - t0
        print(t_0sr1)
        t0 = time.time()
        _, _, _, iter_prox_grad = pg.proximal_gradient(f, gf, x0, 1 / L, 
                                                       l_reg = l, timing = 1, 
                                                       max_iter = max_iter, 
                                                       epsilon = 1e-6)
        t_prox_grad += time.time() - t0
        print(t_prox_grad)
        t0 = time.time()
        _, _, d = spopt.fmin_l_bfgs_b(f_l_bfgs_b, x0_l_bfgs_b, gf_l_bfgs_b, 
                            bounds = bounds, maxiter = max_iter)
        t_l_bfgs_b += time.time() - t0
        print(t_l_bfgs_b)
        print(i)
    
    t_0sr1 /= 5
    t_prox_grad /= 5
    t_l_bfgs_b /= 5
    
    return t_0sr1, t_prox_grad, t_l_bfgs_b, iter_0sr1, iter_prox_grad, d['nit']


if __name__ == "__main__":
    
    import numpy as np
    import scipy.sparse as spa
    import scipy.sparse.linalg as splin
    
    (A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, l, L, 
     fval_l_bfgs_b, xval_l_bfgs_b, max_iter) = initialize_lasso((13, 13, 13), 1, 30)
    x0 *= 0.1
    x0_l_bfgs_b *= 0.1
    
    def f(x):
        temp = A.dot(x) - b
        return 1 / 2 * np.dot(temp.T, temp) + l * np.linalg.norm(x, ord = 1)
    
    def gf(x):
        
        return  A_sq.dot(x) - Ab
        
    def f_l_bfgs_b(x):
        n = len(x)
        temp = A.dot(x[:np.floor(n/2)]) - A.dot(x[np.floor(n/2):]) - b_l_bfgs_b
        return 1/2 * np.dot(temp.T, temp) + l * sum(abs(x))
        
    def gf_l_bfgs_b(x):
        n = len(x)
        temp = A_sq.dot(x[:np.floor(n/2)]) - A_sq.dot(x[np.floor(n/2):])
        return np.concatenate((temp, -temp)).T - Ab_l_bfgs_b + l
    
    fval_0sr1, xval_0sr1, fval_prox_grad, xval_prox_grad = prox_comparison()
    #f_conv_plot(fval_0sr1, fval_prox_grad)
    #x_conv_plot(xval_0sr1, xval_prox_grad)
    t_0sr1, t_prox_grad, t_l_bfgs_b, iter_0sr1, iter_prox_grad, iter_l_bfgs_b = conv_time()
    