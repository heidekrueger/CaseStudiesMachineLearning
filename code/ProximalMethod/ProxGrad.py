# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:48:16 2015

@author: Fin Bauer

This version works as of 26.6.2015 15:52
"""

import numpy as np

def proximal_gradient(f, grad_f, x0, t, **options):
    """
    proximal gradient for l1 regularization
    """
    
    options.setdefault('l_reg', 1)
    
    x_old = x0
    fval = []
    
    for i in range(55):
        
        x_new = prox(x_old - t * grad_f(x_old), t, **options)
        s = x_new - x_old
        
#        t = line_search(f, grad_f, s, x_old, **options)
#        x_new = x_old + t * s
        
        if np.linalg.norm(s) < 1e-8:
            break
        
        x_old = x_new
        fval.append(float(f(x_new)))
        
    return fval
    
def prox(x, t, **options):
    
    return np.maximum(x - t * options['l_reg'], 0) - np.maximum(-x - t * 
                        options['l_reg'], 0)