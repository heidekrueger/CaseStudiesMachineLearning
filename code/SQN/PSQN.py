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
    def __init__(self, options):
        SQN.__init__(self, options)
        self.options['L'] = 1
        self.options['M'] = 1.0
        self.options['r_diff'] = 1.0
        self.options['gamma'] = 0.8
        self.options['tau'] = 2.0
        self.options['tau_min'] = 1e-12
        self.options['tau_max'] = 1e4
        self.options['l_reg'] = 1.0
        self.options['const_step'] = 1.0
        self.options['ls'] = 1.0
        self.options['beta'] = 1.0
        self.options['sr1'] = True
        self.s, self.y = [], []
        self.g_S = None
        self.w_previous = None
        
    def _armijo_rule(self, f, g, x, s, start = 1.0, beta=.5, gamma= 1e-4 ):
        """
        no line search here!
        """
        return(self.options['const_step'])
        
    def _get_search_direction(self, g_S):
            
            t = self.options['const_step']
            if self.options['sr1']:
                    self.g_S = g_S
                    u_H, u_B, d_H, d_B = ProxMeth.compute_sr1_update(self.s, self.y, **self.options)
                    x_new = ProxMeth.compute_proximal(u_H, u_B, d_H, d_B, g_S, self.w.copy(), **self.options)
                    x_new = np.array(x_new.flat)
            else:
                    x_new = prox_grad(self.w - t * g_S(self.w), t, self.options['l_reg'])
            
            self.w_previous = self.w
            return(x_new - self.w)
        
    def _update_correction_pairs(self, g, X, z):
        if self.w_previous is None:
                self.s, self.y = [], []
        else:
                self.s, self.y = self._get_correction_pairs(self.g_S, self.w, self.w_previous)
        return
            