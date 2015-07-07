# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:27:00 2015

@author: Fin Bauer
"""

from __future__ import division

def test_gamma():
    import ProxMeth as pm
    import matplotlib.pyplot as plt
    for i in range(20):
        fval = pm.compute_0sr1(f, gf, x0, gamma = (i+1) / 10)
        plt.plot(range(len(fval)), fval)
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

def read_fval(x):
    
    f_current = f_l_bfgs_b(x)
    fval_l_bfgs_b.append(f_current)
    return

def initialize_lasso(size_A, lam, alpha=0.4, beta=0.7, gamma=0.5, sigma=50):
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
    
    return A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, lam, L
    
def prox_comparison():
    
    import matplotlib.pyplot as plt
    import scipy.optimize as spopt
    import ProxMeth as pm
    import ProxGrad as pg
    from matplotlib2tikz import save as tikz_save
    
    fval_0sr1 = pm.compute_0sr1(f, gf, x0, l_reg = l)
    fval_prox_grad = pg.proximal_gradient(f, gf, x0, 1 / L, l_reg = l)
    spopt.fmin_l_bfgs_b(f_l_bfgs_b, x0_l_bfgs_b, gf_l_bfgs_b, 
                        bounds = bounds, callback = read_fval, maxiter = 300)
    fval_0sr1.insert(0, f(x0))
    fval_prox_grad.insert(0, f(x0))
    fval_l_bfgs_b.insert(0, f_l_bfgs_b(x0_l_bfgs_b))
    line1, = plt.plot(range(len(fval_0sr1)), fval_0sr1, 'r', label = '0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_prox_grad)), fval_prox_grad, 'b', label = 'ProxGrad', lw = 2)
    line3, = plt.plot(range(len(fval_l_bfgs_b)), fval_l_bfgs_b, 'g', label = 'L-BFGS-B', lw = 2)
    plt.xlim([0, 20])
    plt.yscale('log')
    #plt.ylim([1e1, 1e13])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2, line3])
    tikz_save( 'ProxPDE.tikz' );

    return

if __name__ == "__main__":
    
    import numpy as np
    import scipy.sparse as spa
    import scipy.sparse.linalg as splin
    
    A, b, b_l_bfgs_b, A_sq, Ab, Ab_l_bfgs_b, x0, x0_l_bfgs_b, bounds, l, L = initialize_lasso((13, 13, 13), 1)
    fval_l_bfgs_b = []
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
    
    prox_comparison()
    
    #test_gamma()