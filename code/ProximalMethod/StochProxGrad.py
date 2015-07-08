# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:58:03 2015

@author: Fin Bauer
"""

import numpy as np

def proximal_gradient(f, grad_f, x0, t, X, y, **options):
    """
    proximal gradient for l1 regularization
    """
    
    options.setdefault('l_reg', 1)
    options.setdefault('batch_size', 1)
    
    x_old = x0
    N = len(X)
    fval = []
    
    for i in range(100):
        
        batch = np.random.choice(N, options['batch_size'], False)
        y_b = y[batch].reshape(options['batch_size'], 1)
        x_new = prox(x_old - t * grad_f(x_old, X[batch], y_b), t, **options)
        s = x_new - x_old
        
#        t = line_search(f, grad_f, s, x_old, **options)
#        x_new = x_old + t * s
        
        if np.linalg.norm(s) < 1e-8:
            break
        
        x_old = x_new
        fval.append(float(f(x_new, X, y)))
        if i % 50 == 0:
            print(i)
    return fval
    
def prox(x, t, **options):
    
    return np.maximum(x - t * options['l_reg'], 0) - np.maximum(-x - t * 
                        options['l_reg'], 0)