import stochastic_tools
from stochastic_tools import stochastic_gradient
from stochastic_tools import armijo_rule
from SQN import SQN
import numpy as np

import ProxMeth

def prox_grad(x, t, l):
    """ 
    prox operator for simple stochastic proximal gradient descent.
    Will be used if L != 1
    """
    return np.maximum(x - t * l, 0) - np.maximum(-x - t * l, 0)


class PSQN(SQN):
    """
    Proximal Methods SQN
    """
    def __init__(self):
        SQN.__init__(self)
        self.options['L'] = 1.0
        self.options['M'] = 1.0
        self.options['r_diff'] = 1.0
        self.w_previous = None
        self.options['gamma'] = 0.8
        self.options['tau'] = 2.0
        self.options['tau_min'] = 1e-12
        self.options['tau_max'] = 1e4
        self.options['l_reg'] = 1.0
        self.options['const_step'] = 0.01
        self.options['ls'] = 1.0
        self.options['beta'] = 1.0
        
    def _armijo_rule(self, f, g, x, s, start = 1.0, beta=.5, gamma= 1e-4 ):
        return(self.options['const_step'])
        
    def _get_search_direction(self, g_S):
            t = self.options['const_step']
            x_old = self.w
            if self.options['L'] == 1:
                    s = [] if len(self.s) == 0 else self.s[-1].copy()
                    y = [] if len(self.y) == 0 else self.y[-1].copy()
                    
                    u_H, u_B, d_H, d_B = ProxMeth.compute_sr1_update(s, y, **self.options)
                    x_new = ProxMeth.compute_proximal(u_H, u_B, d_H, d_B, g_S, self.w.copy(), **self.options)
                    x_new = np.array(x_new.flat)
            else:
                    x_new = prox_grad(self.w - t * g_S(self.w), t, self.options['l_reg'])
                    
            return(x_new - self.w)
            
            