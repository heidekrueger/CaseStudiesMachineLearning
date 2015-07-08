import numpy as np
from stochastic_tools import sample_batch

class Optimizer:
    """
    Attributes:
    - options: contains everythin to run sqn
    - debug : verbose controler
    - termination_counter : counter
    - iterations : counter of the number of iterations
    - iterator

    Methods:
    - __init__ : builder
    - set_options
    - set_option : to delete ???
    - get_options
    - print_options


    for initialization: Provide one of w1, dim or flat iterator object
    M: Memory-Parameter
    L: Compute Hessian information every Lth step
    beta: start configuration for line search
    batch_size: number of samples to be drawn for gradient evaluation
    batch_size: number of samples to be drawn for Hessian approximation
    max_iter: Terminate after this many steps
    debug: Print progress statements
    """
    def __init__(self, options=None):

        # In case options is not given
        self.options = dict()
        self.options['w1'] = None
        self.options['dim'] = None
        self.options['iterator'] = None
        self.options['max_iter'] = 1e3
        self.options['eps'] = 1e-8
        self.options['r_diff'] = 0.01
        
        self.options['sampleFunction'] = sample_batch
        self.options['M'] = 10
        self.options['L'] = 1
        self.options['N'] = None
        self.options['beta'] = 1
        self.options['batch_size'] = 1
        self.options['batch_size_H'] = 1
        self.options['testinterval'] = 0
        self.options['normalize'] = False
        self.options['updates_per_batch'] = 1
        
        # If options is given
        if options is not None:
                self.set_options(options)
        
        self.debug = False
        self.termination_counter = 0
        self.iterations = 0
        
        self.w = None
        self.f_vals = []
        self.g_norms = []
    
    ############################################
    ############################################
    
    def set_options(self, options):
        '''
        Sets options options attributes to Optimizer object

        INPUTS:
        - self : Optimizer object
        - options : dictionary like containing options to set
        '''
        for key in options:
            if key in self.options:
                self.options[key] = options[key]


    def set_option(self, key, value):
        '''
        Problem here: options is not defined

        => should we delete it ?
        '''
        if key in options:
            self.options[key] = value


    def get_options(self):
        '''
        gets options attributes from Optimizer object

        INPUT:
        - self

        OUTPUT:
        - options attribute, dictionary like
        '''
        return self.options


    def print_options(self):
        '''
        Prints fields and values contained in attribute options of 
        Optimizer object

        INPUT:
        - self
        '''
        for key in self.options:
            print(key)
            print(self.options[key])


    ############################################
    ############################################
        
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
        if all( [w1 == None, dim == None, iterator == None] ):
                w1 = self.options['w1']
                dim = self.options['dim']
                iterator = self.options['iterator']

        assert not all([w1 is None, dim is None, iterator is None]), \
            "Please privide either a starting point or the dimension of the optimization problem!"
            
        if w1 is None and dim is None:
            self.options['iterator'] = iterator
            w1 = stochastic_tools.iter_to_array(self.options['iterator'])
        elif w1 is None:
            w1 = np.ones(dim)
            
        self.options['dim'] = len(w1)
        self.w = w1
        
        print(self.options)
        if self.debug: 
                print(dim)
                print(self.w.shape)
        return


    def get_position(self):
        '''
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
            raise NotImplementedError


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



class StochasticOptimizer(Optimizer):
    """
    Class StochasticOptimizer
    
    Augments the class Optimizer by some stochastic methods
    """
    def __init__(self, options=None):
            
            Optimizer.__init__(self, options)


    def solve_one_step(self, f, g, X=None, z=None, k=1):
            raise NotImplementedError


    def _get_search_direction(self, g_S):
            raise NotImplementedError


    def _draw_sample(self, X=None, z=None, b=None, recursion_depth=1):
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
        
        sample = self.options['sampleFunction']
        N = len(X) if X is not None else self.options['N']
        
        if X is None:
                X_S, z_S = sample(self.w, self.options['N'], b=b)
        else:
                X_S, z_S = sample(self.w, X, z=z, b=b)

        # if empty sample try one more time:
        if len(X_S) == 0 and recursion_depth > 0:
                X_S, z_S = self._draw_sample(X, z=z, b=b,
                                         recursion_depth=recursion_depth-1)

        if self.debug:
                print("sample length: %d, %d" % (len(X_S), len(z_S)))
        return X_S, z_S

