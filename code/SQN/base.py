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
        self.options['testinterval'] = 30
        self.options['normalize'] = False
        self.options['updates_per_batch'] = 1
        
        # If options is given
        if options is not None:
                self.set_options(options)
        
        self.debug = False
        self.termination_counter = 0
        self.iterations = 0
        
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

