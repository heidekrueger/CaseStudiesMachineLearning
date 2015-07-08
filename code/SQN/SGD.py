"""
    Description : Implementation of SQN method
    @author : Roland Halbig / Stefan Heidekrueger
"""

import numpy as np
import itertools
from collections import deque
from statsmodels.stats.diagnostic import lillifors

try: import stochastic_tools
except: import SQN.stochastic_tools as stochastic_tools
try: from stochastic_tools import test_normality
except: from SQN.stochastic_tools import test_normality
"""
TODO: Iterator support not yet tested! Try on Dictionary Learning Problem!
"""

from base import StochasticOptimizer

class SGD(StochasticOptimizer):
    """
    TODO: Two-Loop-Recursion!

    Attributes:
    - s, y: correction pairs
    - w: location
    - L: correction patrameter
    - wbar: mean location of last L steps
    - f_vals: List of intermediate sample evaluations
    - g_norms: List of intermediate gradient lengths
    
    Methods:
    - solve
    - solve_one_step
    - get_position
    
    SOURCE: A Stochastic Quasi-Newton Method for Large-Scale Optimization, Byrd et al. Feb 19 2015, Department of Computer Science University of Colorado
    """

    def __init__(self, options=None):
     
        StochasticOptimizer.__init__(self, options)
        

    def solve_one_step(self, f, g, X=None, z=None, k=1):
            """
            perform one update step
            will update self

            INPUTS:
            - f
            - g
            - X
            - z
            - k
            """
            assert self.w is not None, "Error! weights not initialized!"
            assert X is not None or self.options['sampleFunction'] is not None, \
                "Please provide either a data set or a sampling function"
            
            # Draw sample batch
            X_S, z_S = self._draw_sample(X, z, b=self.options['batch_size'])
            
            # Stochastic functions
            f_S = lambda x: f(x, X_S, z_S)  
            g_S = lambda x: g(x, X_S, z_S)
            
            self.f_vals.append(f_S(self.w))
            self.g_norms.append(np.linalg.norm(g_S(self.w)))

            # perform gradient one or more updates using armijo rule and hessian information
            for i in range(max(1,self.options['updates_per_batch'])):
                    self.w = self._perform_update(f_S, g_S, k)
                    if self.options['normalize']:
                            if self.debug: 
                                    print "Normalizing position"
                                    print np.linalg.norm(self.w)
                            self.w = np.multiply(min(1.0,1.0/np.linalg.norm(self.w)), self.w)
                            
                    
            
            # Check Termination Condition
            if len(X_S) == 0 or self._has_terminated(g_S(self.w), self.w):
                self.termination_counter += 1
            if self.options['testinterval'] > 0 and k % self.options['testinterval'] == 0 and self._is_stationary() and self._get_test_variance() < 0.005:
                    self.termination_counter += 1
                    if self.debug: print("stationary")
            
            return self.w


    def _get_search_direction(self, g_S):
            '''
            INPUTS:
            - g_S: Stochastic gradient evaluated at sample X_S, z_S
            OUTPUT:
            - search_direction : negative gradient (np.array)
            '''
            return -g_S(self.w)


    def _perform_update(self, f_S, g_S, k = None):
            """
            do the gradient updating rule
            INPUTS:
            - f
            - g
            - X
            - z
            OUTPUT: self.w
            """
            # Get search direction
            search_direction = self._get_search_direction(g_S)

            # Line Search
            # TODO
            if k is None:
                    alpha = self._armijo_rule(f_S, g_S, search_direction, start=self.options['beta'], beta=.5, gamma=1e-2)
            else:
                    alpha = self.options['beta']/(k+1.)
            if self.debug: print("step size: %f" % alpha)

            self.w = self.w + np.multiply(alpha, search_direction)
                   
            return self.w


    def _test_normality(self, f, level = 0.01, burn_in = 200):
            if len(f) <= burn_in + 1:
                    return False
            return lillifors(f)[1] > level
    
    def _is_stationary(self):
            """
            stationarity tests
            OUTPUT: True or False
            """
            #phi = self.g_norms
            phi = self.f_vals
            test_len = len(phi) - self.options['testinterval']
            return test_len > 0 and self._test_normality(phi[test_len:], burn_in=1, level=0.05)     
    
    def _get_test_variance(self):
            """
            OUTPUT: Variance estimate of the functionvalue in the testinterval
            """
            test_len = len(self.f_vals) - self.options['testinterval']
            return np.var(self.f_vals[test_len:])
      