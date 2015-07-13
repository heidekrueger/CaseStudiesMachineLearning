# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:58:03 2015

@author: Fin Bauer
"""

import numpy as np

def proximal_gradient(f, grad_f, x0, t, X, y, **options):
    """
    Stochastic proximal gradient method for function of form
    F = f + lambda * h, with f smooth and h the l1-norm (i.e. ||x||_1)

    Arguments
    ---------
    f : f in F
    grad_f : Gradient of f
    x0 : starting point (2D array: n x 1)
    t : fixed step size
    X : training set (2D array: nsamples x nfeatures)
    y : target (2D array: nsamples x 1)

    Options
    -------
    l_reg : lambda in F (regularization parameter)
    max_iter : maximal number of iterations
    epsilon : accuracy of stopping criterion
    timing : set != 0 if algorithm is timed to avoid unnecessary function
             evaluations and prints

    Returns
    -------
    fval : list of f value in each iteration
    x_new : optimal point
    i : number of iterations
    """

    options.setdefault('l_reg', 1)
    options.setdefault('batch_size', 1)
    options.setdefault('max_iter', 1e7)

    x_old = x0
    N = len(X)
    fval = []

    for i in range(options['max_iter']):

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

    return fval, x_new, i

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
