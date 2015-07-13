# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:06:28 2015

@author: Fin Bauer
"""

import numpy as np
from collections import deque


def compute_0sr1(f, grad_f, x0, **options):
    """
    Zero-memory Symmetric Rank 1 Proximal method for function of form
    F = f + lambda * h, with f smooth and h the l1-norm (i.e. ||x||_1)

    Arguments
    ---------
    f : f in F
    grad_f : Gradient of f
    x0 : starting point (2D array: n x 1)

    Options
    -------
    l_reg : lambda in F (regularization parameter)
    max_iter : maximal number of iterations
    epsilon : accuracy of stopping criterion
    gamma : damping parameter for Barzilai-Borwein step size
    tau : diagonal element of H in first iteration
    tau_min, tau_max : limits for Barzilai-Borwein step size
    ls : line-search
    beta : step-size parameter in case line search is used
    timing : set != 0 if algorithm is timed to avoid unnecessary function
             evaluations and prints
    dim : not really an option

    Returns
    -------
    x_new : optimal point
    fval : list of f value in each iteration
    xval : list of iterates of the last 500 iterations
    k : number of iterations
    """

    # set default values for parameters
    options.setdefault('epsilon', 1e-8)
    options.setdefault('gamma', 0.8)
    options.setdefault('tau', 0.01)
    options.setdefault('tau_min', 1e-12)
    options.setdefault('tau_max', 1e4)
    options.setdefault('l_reg', 1)
    options.setdefault('ls', 1)
    options.setdefault('beta', 0.5)
    options.setdefault('max_iter', 1e5)
    options.setdefault('timing', 0)
    n = len(x0)
    options.setdefault('dim', n)
    x0 = x0.reshape((n, 1))
    s = []
    y = []
    x_new = x0.copy()

    p = np.empty((n, 1))
    fval = [float(f(x0))]
    xval = deque([x0], 500)

    for k in range(1, int(options['max_iter']) + 1):

        u_H, u_B, d_H, d_B = compute_sr1_update(s, y, **options)
        temp_x_new = compute_proximal(u_H, u_B, d_H, d_B, grad_f, x_new, **options)

        x_old = x_new
        p = temp_x_new - x_old

        x_new = temp_x_new

        if np.linalg.norm(p) < options['epsilon']: # termination criterion
            break
        #t = line_search(f, h, p, x_old, **options)
        #x_new = x_old + t * p
        #s = t * p
        s = x_new - x_old
        y = grad_f(x_new) - grad_f(x_old)

        if options['timing'] == 0:
            fval.append(float(f(x_new)))
            xval.append(x_new)
            if k % 100 == 0:
                print(k)

    return x_new, fval, xval, k



def compute_sr1_update(s, y, **options):
    """
    Computes Zero-memory Symmetric Rank 1 update

    Arguments
    ---------
    s : x_new - x_old
    y : grad_f(x_new) - grad_f(x_old)

    Returns
    -------
    u_H : update vector for H
    u_B : update vector for B
    d_H : diagonal element for H
    d_B : diagonal element of B
    """

    n = len(s)
    gamma = options['gamma']
    tau = options['tau']
    tau_min = options['tau_min']
    tau_max = options['tau_max']

    if len(s) + len(y) == 0:
        n = options['dim']
        d_H = tau
        u_H = np.zeros((n, 1))
        d_B = 1 / tau
        u_B = np.zeros((n, 1))
    else:
        y.shape = (len(y), 1)
        s.shape = (len(s), 1)
        y_squared = np.dot(y.T, y)
        tau_bb2 = np.dot(s.T, y) / y_squared # Barzilai-Borwein step length
        tau_bb2 = np.median([tau_min, tau_bb2, tau_max]) # Projection
        d_H = gamma * tau_bb2
        d_B = 1 / d_H

        # skip quasi-Newton update
        if (np.dot(s.T, y) - d_H * np.dot(y.T, y) <=
                1e-8 * np.sqrt(y_squared) * np.linalg.norm(s - d_H * y)):
            u_H = np.zeros((n, 1))
            u_B = np.zeros((n, 1))
        else:
            u_H = (s / np.sqrt(np.dot(s.T, y) - d_H * np.dot(y.T, y)) -
                   (d_H * y / np.sqrt(np.dot(s.T, y) - d_H * np.dot(y.T, y))))
            u_B = u_H * d_B / np.sqrt(1 + d_B * np.dot(u_H.T, u_H))

    return u_H, u_B, d_H, d_B



def compute_proximal(u_H, u_B, d_H, d_B, grad, x, **options):
    """
    Calls function-specific subroutines and subsquently computes proximal
    step (s. Corollary 9 and Prop. 10 in paper)
    """
    grad_f = grad(x)

    grad_f.shape = u_B.shape
    x.shape = u_B.shape

    step = d_B * x - u_B * np.dot(u_B.T, x) - grad_f
    alpha = compute_root(step, u_H, d_H, **options)
    proxi = prox(step - alpha / d_H * u_H, **options)
    result = (x - d_H * grad_f - u_H * np.dot(u_H.T, grad_f) -
              (d_H * proxi + u_H * np.dot(u_H.T, proxi)))

    return result



def compute_root(x, u_H, d_H, **options):
    """
    Computes the root of p as in paper
    """

    t = get_transition_points(x, **options)
    trans_points_sorted = sort_transition_points(x, u_H, d_H, t)
    alpha = binary_search(trans_points_sorted, x, u_H, d_H, **options)

    return alpha



def get_transition_points(x, **options):
    """
    returns the transition points t_j for separable h,
    i.e. prox_h(x) = ax + b for t_j <= x <= t_(j+1)
    """

    return options['l_reg'] * np.tile([-1, 1], (len(x), 1))



def sort_transition_points(x, u_H, d_H, t):
    """
    sorts transition points but now of the form d/u*(x-t) from smallest
    to largest
    """

    # exclude indices i for which u_i = 0
    zeros = u_H != 0
    zeros = zeros.reshape(len(zeros),)
    u_H = u_H[zeros]
    t = t[zeros,]
    x = x[zeros]
    u_H = u_H.reshape(len(u_H), 1)
    x = x.reshape(len(x), 1)

    # if u = zero
    if len(u_H) == 0:
        trans_points = np.empty(0,)
    else:
        trans_points = np.sort(x * d_H / u_H - t * d_H / u_H, axis=None)

    return trans_points



def p(alpha, x, u, d, **options):
    """
    p as in paper
    """

    return np.dot(u.T, x) - np.dot(u.T, prox(x - alpha / d * u, **options)) + alpha



def prox(x, **options):
    """
    computes proximal operator for indicator of l_inf-ball
    """

    n = len(x)

    return np.median([-options['l_reg'] * np.ones((n, 1)), x,
                      options['l_reg'] * np.ones((n, 1))], axis=0)


import math
def binary_search(trans_points, x, u, d, **options):
    """
    performs binary search on sorted transition points to obtain root of p
    for separable h
    """

    # no transitions points just a straight line
    if len(trans_points) == 0:
        alpha = 0
    else:
        p_left = p(trans_points[0], x, u, d, **options)
        p_right = p(trans_points[-1], x, u, d, **options)

        # p values of all transition points are below zero
        if np.logical_and(p_left < 0, p_right < 0):
            p_end = p(trans_points[-1] + 1, x, u, d, **options)
            alpha = trans_points[-1] - p_right / (p_end - p_right)
        # p values of all transition points are above zero
        elif np.logical_and(p_left > 0, p_right > 0):
            p_end = p(trans_points[0] - 1, x, u, d, **options)
            alpha = trans_points[-1] - 1 - p_end / (p_left - p_end)
        # normal case
        else:
            left, right = 1, len(trans_points)
            while right - left != 1:
                middle = math.floor(float(left + right)/2.)
                middle = int(middle)
                p_middle = p(trans_points[int(middle - 1)], x, u, d, **options)
                if p_middle == 0:
                    alpha = trans_points[middle - 1]
                    break
                elif p_middle < 0:
                    left = middle
                    p_left = p_middle
                else:
                    right = middle
                    p_right = p_middle
            alpha = (trans_points[left - 1] - p_left *
                     (trans_points[right - 1] - trans_points[left - 1]) /
                     (p_right - p_left))

    return alpha



def line_search(f, h, p, x_old, **options):
    """
    Computes line search factor dependent on chosen line search method
    """

    # step length always equals one
    if options['ls'] == 1:
        t = 1
    # Armijo-type rule
    else:
        t = compute_simple_ls(f, h, p, x_old, **options)

    return t


def compute_simple_ls(f, h, p, x_old, **options):
    """
    simple standard line search
    """

    beta = 1
    F_old = f(x_old) + h(x_old)
    while f(x_old + beta * p) + h(x_old + beta * p) > F_old:
        beta *= options['beta']
        if beta < 1e-20:
            print('break')
            break

    return beta
