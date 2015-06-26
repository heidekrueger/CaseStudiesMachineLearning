# -*- coding: utf-8 -*-
"""
Implementation of Zero-memory Symmetric Rank 1 algorithm to solve min f + h
where f is smooth and h non-smooth
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
    options.setdefault('lambd', 1)

    n = len(x0)
    x0 = x0.reshape((n, 1))
    s = np.zeros((n, 1))
    y = np.zeros((n, 1))
    x_new = x0.copy()

    p = np.empty((n, 1))
    
    #f_x = np.zeros((39,1))
    
    for k in range(1, 300): # make while or itercount later
        
        u_H, u_B, d_H, d_B = compute_sr1_update(s, y, k, **options)
        temp_x_new = compute_proximal(u_H, u_B, d_H, d_B, grad_f, x_new, **options)
        
        x_old = x_new
        p = temp_x_new - x_old
        
        #print(np.sqrt(np.dot(p.T, p)))
        if np.linalg.norm(p) < options['epsilon']: # termination criterion
            break
        
        t = line_search(f, h, y, p, x_old, **options)
        x_new = x_old + t * p
        
        s = t * p
        y = grad_f(x_new) - grad_f(x_old)
        
        print(k)
        #f_x[k-1]=f(x_new)
        print(f(x_new))
    
    return x_new, k, f_x



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
        inter = s - d_H * y # save to reduce computational cost
        
        # skip quasi-Newton update
        if np.dot(inter.T, y) <= 1e-8 * np.sqrt(y_squared) * np.linalg.norm(inter):
            u_H = np.zeros((n, 1))
        else:
            u_H = inter / np.sqrt(np.dot(inter.T, y))
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
    interm = alpha * u_H / d_H
    proxi = prox(step - interm, d_H, **options)
    result = x - d_H * (grad_f - proxi) + u_H * np.dot(u_H.T, 
                (grad_f - proxi))
    
    return result



def compute_root(x, u_H, d_H, **options):
    """
    Computes the root of p as in paper
    """
    
    # root computation for separable case
    if options['algo'] in [1, 2, 3]:
        t = get_transition_points(x, **options)
        trans_points_sorted = sort_transition_points(x, u_H, d_H, t)
        alpha = binary_search(trans_points_sorted, x, u_H, d_H, **options)
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
        diff = x - t
        trans_points = np.sort(diff * d_H / u_H, axis = None)
    
    return trans_points



def p(alpha, u, x, d, **options):
    """
    p as in paper
    """
    
    return np.dot(u.T, x - prox(x - alpha * u / d, d, **options)) + alpha



def prox(x, d, **options):
    """
    computes proximal operators depending on chosen h
    """
    
    # h = l1-norm
    if options['algo'] == 1:
        prox = l1norm_prox(x, d, **options)
    else:
        pass
    
    return prox



def l1norm_prox(x, d, **options):
    """
    computes proximal operator of l1-norm
    """
    
    n = len(x)
    
    return np.median([-np.ones((n,1)), x, np.ones((n,1))], axis = 0)
    
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



def line_search(f, h, y, p, x_old, **options):
    """
    Computes line search factor dependent on chosen line search method
    """
    
    # step length always equals one
    if options['ls'] == 1:
        t = 1
    # Armijo-type rule
    elif options['ls'] == 2:
        t = compute_armijo_ls(f, h, p, x_old, **options)
    # Barzilai-Borwein step size
    elif options['ls'] == 3:
        t = compute_bb_ls(y, p)
    # simple standard line search
    elif options['ls'] == 4:
        t = compute_simple_ls(f, h, p, x_old, **options)
    else: # maybe nonmonotone line search as in Zhang and Hager
        pass
        
    return t



def compute_armijo_ls(f, h, p, x_old, **options):
    """
    Armijo-type rule as in paper of Fukushima and Mine
    """
    
    p_squared = np.dot(p.T,p)
    F_old = f(x_old) + h(x_old)
    beta = 1
    
    while f(x_old + beta * p) + h(x_old + beta * p) > (F_old - options['eta']
            * beta * p_squared):
        beta = beta * options['beta']
        if beta < 1e-8:
            break
            
    return beta



def compute_bb_ls(y, p):
    """
    Barzilai-Borwein step size
    """
    if np.any(y) == False:
        t = 1
    else:
        t = np.dot(p.T, y) / np.dot(p.T, y)
        
    return t



def compute_simple_ls(f, h, p, x_old, **options):
    """
    simple standard line search
    """
    
    beta = 1
    F_old = f(x_old) + h(x_old)
    while f(x_old + beta * p) + h(x_old + beta * p) > F_old:
        beta *= options['beta']
        if beta < 1e-8:
            print(beta)
            break
        
    return beta
        



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso
    #a = 1
    #b = 100
    #rosenbrock = lambda x: (a - (x[0]+1))**2 + b*(x[1]+1 - (x[0]+1)**2)**2
    #rosengrad = lambda x: np.asarray([2*(a-x[0]-1)*(-1) + 2*(x[1]-(x[0]+1)**2)
    #                                    *(-2*(x[1]+1)), 2*(x[1]-(x[0]+1)**2)])
    
    #def f(x):
    #    return x**2
    #def grad_f(x):
    #    return 2*x
        
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
    
    #lars = Lasso(alpha=0.001)
    #lars.fit(A, b)
    #x=lars.coef_
    # x, k = compute_0sr1(f, grad_f, x0, algo=2, lower_b=np.array([0,0]), upper_b=np.array([100,400]))
    
    x, k, f_x = compute_0sr1(z, grad_z, h, x0, ls = 1, lambd = 10)
    #plt.plot(range(1,40), f_x)