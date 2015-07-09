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


import numpy as np
import itertools
from collections import deque
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
        self.options['sampleFunction'] = stochastic_tools.sample_batch
        self.options['M'] = 10
        self.options['L'] = 1
        self.options['N'] = None
        self.options['beta'] = 1
        self.options['batch_size'] = 1
        self.options['batch_size_H'] = 1
        self.options['testinterval'] = 0
        self.options['normalize'] = False
        self.options['normalization'] = None
        
        self.options['updates_per_batch'] = 1
        if options is not None:
                self.set_options(options)

    def solve_one_step(self, f, g, X=None, z=None, k=1):
            """
            perform one update step
            will update self

            INPUTS:
            - f function to be minimized (can be stochastic)
            - g gradient of the function (can be stochastic)
            - X data features
            - z data labels
            - k iteration counter
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
                    self.w = self._perform_update(f_S, g_S)
                    if self.options['normalize']:
                            if self.options['normalization'] is None:
                                    print("Warning:Normalization function is not provided; Using L2-normalization")
                                    self.w = np.multiply(1.0/np.linalg.norm(self.w), self.w)
                            else:
                                    self.w = self.options['normalization'](self.w)
                            if self.debug: 
                                    print "Normalizing position"
                            
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
            """
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
      
"""
    Description : Implementation of SQN method
    @author : Roland Halbig / Stefan Heidekrueger
"""


class SQN(SGD):
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
        
        SGD.__init__(self, options)

        self.s, self.y = deque(), deque()
        self.wbar = None
        self.wbar_previous = None
        self.H = None

    def solve_one_step(self, f, g, X=None, z=None, k=1):
        """
            perform one update step
            will update self

            INPUTS:
            - f function to be minimized (can be stochastic)
            - g gradient of the function (can be stochastic)
            - X data features
            - z data labels
            - k iteration counter
            """
        self.w = SGD.solve_one_step(self, f, g, X, z, k)
        # update wbar and get new correction pairs
        self.wbar = self.w if self.wbar is None else self.wbar + self.w 
        if self.options['batch_size_H'] > 0 and self.options['L'] > 0 and k % self.options['L'] == 0:
            self.wbar /= float(self.options['L'])
            if self.wbar_previous is not None:
                if self.debug: print("HESSE")
                self._update_correction_pairs(g, X, z)
                self.H = self._get_H()
            self.wbar_previous = self.wbar
            self.wbar = np.zeros(self.options['dim'])

        return self.w

    # Determine search direction
    def _get_search_direction(self, g_S):
        '''
        Determines search direction

        INPUTS:
        - g_S: Stochastic gradient evaluated at sample X_S, z_S

        OUTPUT:
        - search_direction : np.array
        '''
        if len(self.y) < 2:
            search_direction = -g_S(self.w)
        else:
            # search_direction = self._two_loop_recursion(g_S(self.w))
            if self.H is None:
                self.H = self._get_H()
            search_direction = -self.H.dot(g_S(self.w))
        if self.debug:
            print("Direction:", search_direction.T)
        return search_direction

    def _get_correction_pairs(self, g_S, w, w_previous):
        """
        Perlmutters Trick:
        https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
        H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
        r = 1e-2
        y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r

        INPUTS:
        - g_S stochastic subsample gradient 
        - w
        - w_previous

        OUTPUTS:
        - s_t
        - y_t
        """
        r = self.options['r_diff']
        s_t = w - w_previous
        s_t = np.multiply(r, s_t)
        y_t = (g_S(w) - g_S(w - s_t)) / (r)

        return s_t, y_t

    def _update_correction_pairs(self, g, X, z):
        """
        returns correction pairs s,y
        TODO: replace explicit stochastic gradient

        INPUTS:
        - self
        - g gradient of the problem
        - X
        - z

        """
        
        # draw hessian sample and get the corresponding stochastic gradient
        X_SH, y_SH = self._draw_sample(X, z, b=self.options['batch_size_H'])
        g_SH = lambda x: g(x, X_SH, y_SH)

        s_t, y_t = self._get_correction_pairs(g_SH,
                                              self.wbar,
                                              self.wbar_previous)

        if self.debug:
            print("correction:")

        if abs(y_t).sum() != 0:
            self.s.append(s_t)
            self.y.append(y_t)
        else:
            print("PROBLEM! zero y")

        if len(self.s) > self.options['M']:
            self.s.popleft()
            self.y.popleft()

        if self.debug:
            print("Length s, y: %d, %d" % (len(self.s), len(self.y)))
        return


    def _get_H(self, debug=False):
        """
        returns H_t as defined in algorithm 2
        TODO: Two-Loop-Recursion
        """
        I = np.identity(len(self.w))
        
        if min(len(self.s), len(self.y)) == 0:
                print "Warning: No second order information used!"
                return I
            
        assert len(self.s) > 0, "s cannot be empty."
        assert len(self.s) == len(self.y), "s and y must have same length"
        assert self.s[0].shape == self.y[0].shape, \
            "s and y must have same shape"
        assert abs(self.y[-1]).sum() != 0, "latest y entry cannot be 0!"
        assert 1/np.inner(self.y[-1], self.s[-1]) != 0, "!"

        # H = (s_t^T y_t^T)/||y_t||^2 * I

        # For now: Standard L-BFGS update
        # TODO: Two-Loop Recursion
        # TODO: Hardcode I each time to save memory. (Or sparse???)
        I = np.identity(len(self.s[0]))
        H = np.dot((np.inner(self.s[-1], self.y[-1]) / np.inner(self.y[-1],
                   self.y[-1])), I)

        for (s_j, y_j) in itertools.izip(self.s, self.y):
            rho = 1/np.inner(y_j, s_j)
            H = (I - rho * np.outer(s_j, y_j)).dot(H).dot(I - rho * np.outer(y_j, s_j))
            H += rho * np.outer(s_j, s_j)

        return H

    def _two_loop_recursion(self, grad):
        """
        TODO: Description two loop recursion and wikipedia link
        TODO: Check and TEST!!
        returns:
        z = H_k g_k

        Might have a problem with this function : s not defined
        """
        if min(len(self.s), len(self.y)) == 0:
                print "Warning: No second order information used!"
                return grad
        
        assert len(s) == len(y), "s and y must have same length"
        assert s[0].shape == y[0].shape, "s and y must have same shape"
        assert abs(y[-1]).sum() != 0, "latest y entry cannot be 0!"
        assert 1/np.inner(y[-1], s[-1]) != 0, "!"
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = grad
        rho = 1./ np.inner(y[-1], s[-1])
        a = []

        for j in range(len(s)):
            a.append( rho * np.inner(s[j], q))
            q = q - np.multiply(a[-1], y[j])

        H_k = np.inner(y[-2], s[-2]) / np.inner(y[-2], y[-2])
        z = np.multiply(H_k, q)

        for j in reversed(range(len(s))):
            b_j = rho * np.inner(y[j], z)
            q = q - np.multiply(a[j] - b_j, s[j])

        return z
