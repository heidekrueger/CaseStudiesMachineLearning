# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:06:28 2015

@author: Fin Bauer
"""

import numpy as np


def compute_0sr1(f, grad_f, h, x0, **options):
    """
    Main function for Zero-memory Symmetric Rank 1 algorithm
    Input-Arguments:
    f: smooth part of F = f + h
    grad_f: Gradient of f
    x0: starting value
    options:...
    """
    
    # set default values for parameters
    options.setdefault('epsilon', 1e-8)
    options.setdefault('gamma', 0.8)
    options.setdefault('tau', 2)
    options.setdefault('tau_min', 1e-12)
    options.setdefault('tau_max', 1e4)
    options.setdefault('l', 1)
    options.setdefault('ls', 1)
    options.setdefault('beta', 0.5)

    n = len(x0)
    x0 = x0.reshape((n, 1))
    s = np.zeros((n, 1))
    y = np.zeros((n, 1))
    x_new = x0.copy()

    p = np.empty((n, 1))
    
    for k in range(1, 5000): # make while or itercount later
        
        u_H, u_B, d_H, d_B = compute_sr1_update(s, y, k, **options)
        temp_x_new = compute_proximal(u_H, u_B, d_H, d_B, grad_f, x_new, **options)
        
        x_old = x_new
        p = temp_x_new - x_old
        
        if np.linalg.norm(p) < options['epsilon']: # termination criterion
            break
        x_new = temp_x_new
        #t = line_search(f, h, p, x_old, **options)
        #x_new = x_old + t * p
        #s = t * p
        s = x_new - x_old
        y = grad_f(x_new) - grad_f(x_old)
        
        #if k % 50 == 0:
        print(f(x_new) + h(x_new))
    
    return x_new, k



def compute_sr1_update(s, y, k, **options):
    """
    Computes Zero-memory Symmetric Rank 1 update
    Input-Arguments:
    s: x_new - x_old
    y: grad_f(x_new) - grad_f(x_old)
    k: iteration count
    
    Returns:
    B: B_k = H_k^-1
    H: Hessian approximation
    u: rank-1 update vector
    d: diagonal-element of B
    """
    
    n = len(s)
    gamma = options['gamma']
    tau = options['tau']
    tau_min = options['tau_min']
    tau_max = options['tau_max']
    
    if k == 1:
        d_H = tau
        u_H = np.zeros((n, 1))
        d_B = 1 / tau
        u_B = np.zeros((n, 1))
    else:
        y_squared = np.dot(y.T, y)
        tau_bb2 = np.dot(s.T, y) / y_squared # Barzilai-Borwein step length
        tau_bb2 = np.median([tau_min, tau_bb2, tau_max]) # Projection
        d_H = gamma * tau_bb2
        d_B = 1 / d_H
        
        # skip quasi-Newton update
        if np.dot(s.T, y) - d_H * np.dot(y.T, y) <= (1e-8 * np.sqrt(y_squared)
                            * np.linalg.norm(s - d_H * y)):
            u_H = np.zeros((n, 1))
            u_B = np.zeros((n, 1))
        else:
            u_H = s / np.sqrt(np.dot(s.T, y) - d_H * np.dot(y.T, y)) - (d_H *
                    y / np.sqrt(np.dot(s.T, y) - d_H * np.dot(y.T, y)))
            u_B = u_H * d_B / np.sqrt(1 + d_B * np.dot(u_H.T, u_H))
            
    return u_H, u_B, d_H, d_B



def compute_proximal(u_H, u_B, d_H, d_B, grad_f, x, **options):
    """
    Calls function-specific subroutines and subsquently computes proximal
    step (s. Corollary 9 and Prop. 10 in paper)
    """
    
    grad_f = grad_f(x)
    step = d_B * x - u_B * np.dot(u_B.T, x) - grad_f
    alpha = compute_root(step, u_H, d_H, **options)
    proxi = prox(step - alpha / d_H * u_H, **options)
    result = x - d_H * grad_f - u_H * np.dot(u_H.T, grad_f) - (
             d_H * proxi + u_H * np.dot(u_H.T, proxi))
    
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
    
    return options['l'] * np.tile([-1, 1], (len(x), 1))



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
        trans_points = np.sort(x * d_H / u_H - t * d_H / u_H, axis = None)
    
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
    
    return np.median([-options['l'] * np.ones((n,1)), x, options['l'] * 
                    np.ones((n,1))], axis = 0)
    


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
            l, r = 1, len(trans_points)
            while r-l != 1:
                m = np.floor(1 / 2 * (l + r))
                p_middle = p(trans_points[m - 1], x, u, d, **options)
                if p_middle == 0:
                    alpha = trans_points[m - 1]
                    break
                elif p_middle < 0:
                    l = m
                    p_left = p_middle
                else:
                    r = m
                    p_right = p_middle
            alpha = trans_points[l - 1] - p_left * (trans_points[r - 1] - 
                    trans_points[l - 1]) / (p_right - p_left)
            
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




if __name__ == "__main__":
        
    A = np.random.normal(size = (1500, 3000))
    b = np.random.normal(size = (1500, 1))
    A_sq = np.dot(A.T, A)
    Ab = np.dot(A.T, b)

    def z(x):
    
        temp = np.dot(A, x) - b
    
        return 1 / 2 * np.dot(temp.T, temp)
    
    def grad_z(x):
        
        return  np.dot(A_sq, x) - Ab
        
    def h(x):
        
        return np.linalg.norm(x, ord = 1)
        
    x0 = np.ones((3000,1))
    
    x, k = compute_0sr1(z, grad_z, h, x0, l = 10)