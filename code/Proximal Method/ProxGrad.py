# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:48:16 2015

@author: Fin Bauer

This version works as of 26.6.2015 15:52
"""

import numpy as np

def proximal_gradient(f, grad_f, h, x0, t, **options):
    """
    proximal gradient for l1 regularization
    """
    
    options.setdefault('l', 1)
    
    x_old = x0
    
    for i in range(10000):
        
        x_new = prox(x_old - t * grad_f(x_old), t, **options)
        s = x_new - x_old
        
#        t = line_search(f, grad_f, s, x_old, **options)
#        x_new = x_old + t * s
        
        if np.linalg.norm(s) < 1e-8:
            break
        
        x_old = x_new
    
    return x_new, i
    
def prox(x, t, **options):
    
    return np.maximum(x - t * options['l'], 0) - np.maximum(-x - t * 
                        options['l'], 0)


if __name__ == "__main__":
    from scipy.linalg import eigvals
    from time import time
    A = np.random.normal(size = (150, 300))
    b = np.random.normal(size = (150, 1))
    A_sq = np.dot(A.T, A)
    Ab = np.dot(A.T, b)

    def z(x):
    
        temp = np.dot(A, x) - b
    
        return 1 / 2 * np.dot(temp.T, temp)
    
    def grad_z(x):
        
        return  np.dot(A_sq, x) - Ab
    
    def h(x):
        
        return np.linalg.norm(x, ord = 1)
        
    print "lol"
    x0 = np.ones((300,1))
    L = eigvals(A_sq).real.max()
    
    t0 = time()
    x, i = proximal_gradient(z, grad_z, h, x0, 1 / L, l = 10)
    print(time() - t0)