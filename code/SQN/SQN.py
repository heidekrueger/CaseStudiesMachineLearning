"""
    Description : Implementation of SQN method
    @author : Roland Halbig / Stefan Heidekrueger
"""

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


def solveSQN(f, g, X, z=None, w1=None, dim=None, iterator=None, M=10, L=1.0,
             beta=1, batch_size=1, batch_size_H=1, max_iter=1e4, debug=False,
             sampleFunction=stochastic_tools.sample_batch):
    '''
    function wrapper for SQN class. Used for backward compatibility.
    '''

    sqn = SQN()
    sqn.debug = debug
    sqn.set_options({'w1': w1,
                     'dim': dim,
                     'iterator': iterator,
                     'M': M,
                     'L': L,
                     'beta': beta,
                     'batch_size': batch_size,
                     'batch_size_H': batch_size_H,
                     'max_iter': max_iter,
                     'sampleFunction': sampleFunction})

    return sqn.solve(f, g, X, z)

from base import Optimizer

class SQN(Optimizer):
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
        
        Optimizer.__init__(self, options)

        self.s, self.y = deque(), deque()
        self.w = None
        self.wbar = None
        self.wbar_previous = None
        self.f_vals = []
        self.g_norms = []
                
        self.iterator = None
        self.H = None
                        
        err_mes1 = "Memory Parameter M must be a positive integer!"
        assert self.options['M'] > 0, err_mes1


    def set_start(self, w1=None, dim=None, iterator=None):
        """
        Set start point of the optimization using numpy array, dim or
        flat.iterator object.

        INPUTS:
        - w1: Start position
        - dim: If only a dimension is given, a zero start point will be used
        - iterator: An iterator can be given in order to save memory. 
        TODO: Not tested!
        OUTPUT: -
        """
        print(self.options)
        if all( [w1 == None, dim == None, iterator == None] ):
                w1 = self.options['w1']
                dim = self.options['dim']
                iterator = self.options['iterator']
                
        err_mes1 = "Memory Parameter M must be a positive integer!"
        assert self.options['M'] > 0, err_mes1

        # start point
        err_mes2 = "Please privide either a starting point or the dimension of the optimization problem!"
        assert w1 is not None or dim is not None or iterator is not None, err_mes2
            
        if self.debug: print(dim)

        if w1 is None and dim is None:
            self.options['iterator'] = iterator
            w1 = stochastic_tools.iter_to_array(self.options['iterator'])
        elif w1 is None:
            w1 = np.ones(dim)
            
        self.options['dim'] = len(w1)
        
        # initialize
        self.w = w1
        self.wbar = np.zeros(self.w.shape)
        self.wbar_previous = None
        self.s, self.y = deque(), deque()
        if self.debug:
            print(self.w.shape)
        return

    def get_position(self):
        '''
        return position

        INPUT: -

        OUTPUT:
        w : position
        '''
        return self.w

    def solve(self, f, g, X=None, z=None):
        """
        Parameters:
            f  := f(omega, x, z), loss function for a complete sample batch
            g := g(omega, X, z) stochastic gradient of the sample
            
            X: list of nFeatures numpy column arrays of Data
            z: list of nSamples integer labels
        """

        assert X is not None or self.options['sampleFunction'] is not None, \
            "Please provide either a data set or a sampling function"

        self.set_start()

        for k in itertools.count():

            if self.debug: print("Iteration %d" % k)

            self.solve_one_step(f, g, X, z, k)
            
            if k > self.options['max_iter'] or self.termination_counter > 4:
                self.iterations = k
                break

        if self.iterations < self.options['max_iter']:
            print("Terminated successfully!")
        print("Iterations:\t\t%d" % self.iterations)

        # id an iterator was used, write the result into it
        if self.options['iterator'] is not None:
            stochastic_tools.set_iter_values(self.options['iterator'], self.w)
            return iterator
        else:
            return self.w

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
        
        # Draw sample batch
        if X is None:
                X_S, z_S = self._draw_sample(self.options['N'], \
                                            b=self.options['batch_size'])
        else:
                X_S, z_S = self._draw_sample(X, z, b=self.options['batch_size'])
        
        # Stochastic functions
        f_S = lambda x: f(x, X_S, z_S)  
        g_S = lambda x: g(x, X_S, z_S)

        # perform gradient one or more updates using armijo rule and hessian information
        for i in range(max(1,self.options['updates_per_batch'])):
                self.w = self._perform_update(f_S, g_S)
        
        # Check Termination Condition
        if len(X_S) == 0 or self._has_terminated(g_S(self.w), self.w):
            self.termination_counter += 1
        if k % self.options['testinterval'] == 0 and self._is_stationary() and self._get_test_variance() < 0.005:
                self.termination_counter += 1
                if self.debug: print("stationary")
                    
        # update wbar and get new correction pairs
        self.wbar += self.w
        if k % self.options['L'] == 0:
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
            # search_direction = -self._two_loop_recursion(g_S)
            if self.H is None:
                self.H = self._get_H()
            search_direction = -self.H.dot(g_S(self.w))
        if self.debug:
            print("Direction:", search_direction.T)
        return search_direction

    def _armijo_rule(self, f, g, s, start=1.0, beta=.5, gamma=1e-4):
        """
        Determines the armijo-rule step size alpha for approximating
        line search min f(x+omega*s)

        Parameters:
            f: objective function
            g: gradient
            x:= x_k
            s:= x_k search direction
            beta, gamma: parameters of rule
        TODO: Reference Source??
        """
        candidate = start
        if self.debug: print "armijo"
        fw = f(self.w)
        rhs = gamma * np.inner(g(self.w), s)
        while candidate > 1e-4 and (f(self.w + np.multiply(candidate, s)) - fw > candidate * rhs):

            candidate *= beta

        return candidate

    # Calculate gradient and perform update
    def _perform_update(self, f_S, g_S):
        """
        do the gradient updating rule

        INPUTS:
        - f
        - g
        - X
        - z

        """
        # Get search direction
        search_direction = self._get_search_direction(g_S)

        # Line Search
        alpha = self._armijo_rule(f_S, g_S, search_direction,
                                  start=self.options['beta'],
                                  beta=.5, gamma=1e-2)

        alpha = max([alpha, 1e-5])
        if self.debug:
            print("step size: %f" % alpha)

        # Update
        self.w = self.w + np.multiply(alpha, search_direction)
        
        if self.options['normalize']:
                self.w = np.multiply(1.0/max(1.0, np.linalg.norm(self.w)), self.w)
                
        # Store information
        self.f_vals.append(f_S(self.w))
        self.g_norms.append(np.linalg.norm(g_S(self.w)))

        return self.w

    def _get_correction_pairs(self, g_S, w, w_previous):
        """
        Perlmutters Trick:
        https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
        H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
        r = 1e-2
        y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r

        INPUTS:
        - g_S
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
        - g
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

    def _draw_sample(self, X, z=None, b=None, recursion_depth=1):
        """
        Draw sample from smaple function. Recurse if empty sample was drawn.

        INPUTS:
        - self
        - X
        - z
        - b
        - recursion_depth

        OUTPUTS:
        - X_S
        - z_S
        """

        if b is None:
            b = self.options['batch_size']

        if X is None and self.options['N'] is None:
            X_S, z_S = self.options['sampleFunction'](self.w,
                                                      self.options['N'], b=b)

        elif X is None and self.options['N'] is not None:
            X_S, z_S = self.options['sampleFunction'](self.w,
                                                      self.options['N'], b=b)

        else:
            X_S, z_S = self.options['sampleFunction'](self.w, X, z=z, b=b)

        # if empty sample try one more time:
        if len(X_S) == 0 and recursion_depth > 0:
            X_S, z_S = self._draw_sample(X, z=z, b=b,
                                         recursion_depth=recursion_depth-1)

        if self.debug:
            print("sample length: %d, %d" % (len(X_S), len(z_S)))
        return X_S, z_S


    def _get_H(self, debug=False):
        """
        returns H_t as defined in algorithm 2
        TODO: Two-Loop-Recursion
        """

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

    def _two_loop_recursion(self, g_S):
        """
        TODO: Description two loop recursion and wikipedia link
        TODO: Check and TEST!!
        returns:
        z = H_k g_k
        """

        """
        Might have a problem with this function : s not defined
        """
        assert len(s) > 0, "s cannot be empty."
        assert len(s) == len(y), "s and y must have same length"
        assert s[0].shape == y[0].shape, "s and y must have same shape"
        assert abs(y[-1]).sum() != 0, "latest y entry cannot be 0!"
        assert 1/np.inner(y[-1], s[-1]) != 0, "!"
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = g_S(self.w)
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

    def _is_stationary(self):
            """
            stationarity tests
            OUTPUT: True or False
            """
            test_len = len(self.f_vals) - self.options['testinterval']
            return test_len > 0 and test_normality(self.f_vals[test_len:], burn_in=1, level=0.05)     
    
    def _get_test_variance(self):
            """
            OUTPUT: Variance estimate of the functionvalue in the testinterval
            """
            test_len = len(self.f_vals) - self.options['testinterval']
            return np.var(self.f_vals[test_len:])
        
    def _has_terminated(self, grad, w):
        """
        Checks whether the algorithm has terminated

        Parameters:
            grad: gradient
            w: current variable
        """
        
        if self.debug:
            print("Check termination")
            print("len grad: %f" % np.linalg.norm(grad))

        if len(grad) > 0 and np.linalg.norm(grad) < self.options['eps']:
            return True
        else:
            return False
