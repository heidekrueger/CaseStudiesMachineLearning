# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:48:16 2015

@author: Fin Bauer

This version works as of 26.6.2015 15:52
"""

import numpy as np
from collections import deque

def proximal_gradient(f, grad_f, x0, t, **options):
    """
    proximal gradient for l1 regularization
    """
    
    options.setdefault('l_reg', 1)
    options.setdefault('max_iter', 1e6)
    options.setdefault('timing', 0)
    options.setdefault('epsilon', 1e-8)
    
    x_old = x0
    fval = [f(x0)]
    xval = deque([x0], 500)
    
    for i in range(int(options['max_iter'])):
        
        x_new = prox(x_old - t * grad_f(x_old), t, **options)
        s = x_new - x_old
        
#        t = line_search(f, grad_f, s, x_old, **options)
#        x_new = x_old + t * s
        
        if np.linalg.norm(s) < options['epsilon']:
            break
        
        x_old = x_new
        
        if options['timing'] == 0:
            fval.append(float(f(x_new)))
            xval.append(x_new)
            if i % 10000 == 0:
                print(i)
        
    return x_new, fval, xval, i
    
def prox(x, t, **options):
    
    return np.maximum(x - t * options['l_reg'], 0) - np.maximum(-x - t * 
                        options['l_reg'], 0)