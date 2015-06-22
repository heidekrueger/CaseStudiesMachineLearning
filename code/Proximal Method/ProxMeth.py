# -*- coding: utf-8 -*-
"""
Implementation of Zero-memory Symmetric Rank 1 algorithm to solve min f + h
where f is smooth and h non-smooth
"""

import numpy as np
import scipy as sp


def compute_0sr1(f, grad_f, x0, **options):
    """
    Main function for Zero-memory Symmetric Rank 1 algorithm
    Input-Arguments:
    f: smooth part of F = f + h
    grad_f: Gradient of f
    x0: starting value
    options:...
    """
    
    # set default values for parameters
    options.setdefault('algo', 1)
    options.setdefault('epsilon', 1e-8)
    options.setdefault('gamma', 0.8)
    options.setdefault('tau', 2)
    options.setdefault('tau_min', 1e-8)
    options.setdefault('tau_max', 1e4)
    options.setdefault('lower_b', None)
    options.setdefault('upper_b', None)
    options.setdefault('ls', 1)
    options.setdefault('beta', 0.5)
    options.setdefault('eta', 1)

    n = len(x0)
    s = np.empty(n,)
    y = np.empty(n,)
    x_new = x0
    
    for k in range(1, 100000): # make while or itercount later
        
        B, H, u, d = compute_sr1_update(s, y, k, **options)
        temp_x_new = compute_proximal(B, H, u, d, grad_f, x_new, **options)
        
        x_old = x_new
        s = temp_x_new - x_old
        
        if np.linalg.norm(s) < options['epsilon']: # termination criterion
            break
        
        t = line_search(f, s, y, temp_x_new, x_old, **options)
        x_new = x_old + t * s
        
        y = grad_f(x_new) - grad_f(x_old)
    
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
        H = sp.sparse.diags(tau, 0, shape = (n, n), format = "csr")
        B = sp.sparse.diags(1 / tau, 0, shape = (n, n), format = "csr")
        u = np.zeros(n,)
        d = 1 / tau
    else:
        y_squared = np.dot(y, y)
        tau_bb2 = np.dot(s, y) / y_squared # Barzilai-Borwein step length
        tau_bb2 = np.median([tau_min, tau_bb2, tau_max]) # Projection
        bla = np.empty(n,)
        bla.fill(gamma * tau_bb2)
        H = np.diag(bla, 0)
        B = np.diag(1 / bla, 0)
        d = 1 / (gamma * tau_bb2)
        inter = s - H.dot(y) # save to reduce computational cost
        
        # skip quasi-Newton update
        if np.dot(inter, y) <= 1e-8 * np.sqrt(y_squared) * np.sqrt(
            np.dot(inter, inter)):
            u = np.zeros(n,)
        else:
            u = inter / np.sqrt(np.dot(inter, y))
            H = H + np.outer(u, u)
            B = B + np.outer(u, u) / gamma**2 / tau_bb2**2 / (1 + np.dot(u, u) 
                / gamma / tau_bb2) # Sherman-Morison formula
                
    return B, H, u, d



def compute_proximal(B, H, u, d, grad_f, x, **options):
    """
    Calls function-specific subroutines and subsquently computes proximal
    step (s. Corollary 9 and Prop. 10 in paper)
    """
    step = x - H.dot(grad_f(x))
    alpha = compute_root(step, u, d, **options)
    
    return prox(step - alpha * u / d, d, **options)



def compute_root(x, u, d, **options):
    """
    Computes the root of p as in paper
    """
    
    # root computation for separable case
    if options['algo'] in [1, 2, 3]:
        t = get_transition_points(x, **options)
        trans_points_sorted = sort_transition_points(x, u, d, t)
        alpha = binary_search(trans_points_sorted, x, u, d, **options)
    # root computation for non-separable
    else:
        pass 
        
    return alpha



def get_transition_points(x, **options):
    """
    returns the transition points t_j for separable h,
    i.e. prox_h(x) = ax + b for t_j <= x <= t_(j+1)
    """
    
    n = len(x)
    
    # transition points for h = l1-norm and l_inf-ball
    if options['algo'] in [1, 3]:
        t = np.tile([-1, 1], (n, 1))
    # transition points for h = box constraint or positivity constraint
    elif options['algo'] == 2:
        t = np.array([options['lower_b'], options['upper_b']]).T
    else:
        pass
    
    return t



def sort_transition_points(x, u, d, t):
    """
    sorts transition points but now of the form d/u*(x-t) from smallest 
    to largest
    """
    
    # exclude indices i for which u_i = 0
    u = u[u != 0]
    t = t[u != 0]
    x = x[u != 0]
    
    # if u = zero
    if len(u) == 0:
        trans_points = np.empty(0,)
    else:
        diff = x.reshape(len(x),1) - t
        trans_points = np.sort(diff.T * d / u, axis = None)
    
    return trans_points



def p(alpha, u, x, d, **options):
    """
    p as in paper
    """
    
    return np.dot(u, x - prox(x - alpha * u / d, d, **options)) + alpha



def prox(x, d, **options):
    """
    computes proximal operators depending on chosen h
    """
    
    # h = l1-norm
    if options['algo'] == 1:
        prox = l1norm_prox(x, d)
    # h = box constraint or positivity constraint
    elif options['algo'] == 2:
        prox = box_constraint_prox(x, **options)
    # h = l_inf-ball
    elif options['algo'] == 3:
        prox = linfball_prox(x)
    else:
        pass
    
    return prox



def l1norm_prox(x, d):
    """
    computes proximal operator of l1-norm
    """
    
    return np.maximum(x - 1 / d, 0) - np.maximum(-x - 1 / d, 0)



def box_constraint_prox(x, **options):
    """
    computes proximal operator for box constraint and positivity constraint
    """
        
    return np.median([options['lower_b'], x, options['upper_b']], axis = 0)



def linfball_prox(x):
    """
    computes proximal operator for l_inf-ball
    """
    
    n = len(x)
    
    return np.median([-np.ones(n), x, np.ones(n)], axis = 0)
    


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



def line_search(f, s, y, x_old, x_new, **options):
    """
    Computes line search factor dependent on chosen line search method
    """
    
    # step length always equals one
    if options['ls'] == 1:
        t = 1
    # Armijo-type rule
    elif options['ls'] == 2:
        t = compute_armijo_ls(f, x_old, x_new, **options)
    # Barzilai-Borwein step size
    elif options['ls'] == 3:
        t = compute_bb_ls(s, y)
    else: # maybe nonmonotone line search as in Zhang and Hager
        pass
        
    return t



def compute_armijo_ls(f, h, x_new, x_old, **options):
    """
    Armijo-type rule as in paper of Fukushima and Mine
    """
    
    d = x_new - x_old
    d_squared = np.dot(d,d)
    F_old = f(x_old) + h(x_old)
    beta = 1
    
    while f(x_old + beta * d) + h(x_old + beta * d) > (F_old - options['eta']
            * beta * d_squared):
        beta = beta * options['beta']
            
    return beta



def compute_bb_ls(s, y):
    """
    Barzilai-Borwein step size
    """
    if len(s) == 0:
        t = 1
    else:
        t = np.dot(s, y) / np.dot(y, y)
        
    return t




if __name__ == "__main__":
    
    #a = 1
    #b = 100
    #rosenbrock = lambda x: (a - (x[0]+1))**2 + b*(x[1]+1 - (x[0]+1)**2)**2
    #rosengrad = lambda x: np.asarray([2*(a-x[0]-1)*(-1) + 2*(x[1]-(x[0]+1)**2)
    #                                    *(-2*(x[1]+1)), 2*(x[1]-(x[0]+1)**2)])
    
    def f(x):
        return x**2
    def grad_f(x):
        return 2*x
        
    x0 = np.array([13,4])
    
    # x, k = compute_0sr1(f, grad_f, x0, algo=2, lower_b=np.array([0,0]), upper_b=np.array([100,400]))
    
    x, k = compute_0sr1(f, grad_f, x0)
    
    print(x)
    print(k)