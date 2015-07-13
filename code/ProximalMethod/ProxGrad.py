# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:48:16 2015

@author: Fin Bauer
"""

import numpy as np
from collections import deque

def proximal_gradient(f, grad_f, x0, t, **options):
    """
    Proximal gradient method for function of form F = f + lambda * h, with f
    smooth and h the l1-norm (i.e. ||x||_1)

    Arguments
    ---------
    f : f in F
    grad_f : Gradient of f
    x0 : starting point (2D array: n x 1)
    t : fixed step size

    Options
    -------
    l_reg : lambda in F (regularization parameter)
    max_iter : maximal number of iterations
    epsilon : accuracy of stopping criterion
    timing : set != 0 if algorithm is timed to avoid unnecessary function
             evaluations and prints

    Returns
    -------
    x_new : optimal point
    fval : list of f value in each iteration
    xval : list of iterates of the last 500 iterations
    i : number of iterations
    """

    options.setdefault('l_reg', 1)
    options.setdefault('max_iter', 1e6)
    options.setdefault('epsilon', 1e-8)
    options.setdefault('timing', 0)

    x_old = x0
    fval = [f(x0)]
    xval = deque([x0], 500)

    for i in range(int(options['max_iter'])):

        x_new = prox(x_old - t * grad_f(x_old), t, **options)
        s = x_new - x_old

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
    """
    Proximity operator for t * lambda * ||x||_1

    Arguments
    ---------
    x : point where proximity operator is evaluated
    t : step size

    Returns
    -------
    Result of proximity operator
    """

    return (np.maximum(x - t * options['l_reg'], 0) -
            np.maximum(-x - t * options['l_reg'], 0))
