import stochastic_tools
from stochastic_tools import stochastic_gradient
from stochastic_tools import armijo_rule
from SQN import SQN
import numpy as np

def prox(x, t, l):
    return np.maximum(x - t * l, 0) - np.maximum(-x - t * l, 0)

class PSQN(SQN):
	"""
	Proximal Methods SQN
	"""
	def __init__(self):
		SQN.__init__(self)
		self.options['L'] = self.options['max_iter'] * 100
		self.options['M'] = 1
		self.w_previous = None
		# TODO: Add further options
	
	def _armijo_rule(self, f, g, x, s, start = 1.0, beta=.5, gamma= 1e-4 ):
		return(1.0)
	    
	def _get_search_direction(self, g_S):
		x_old = self.w
		t = 0.001
		l = 1.0
		x_new = prox(self.w - t * g_S(self.w), t, l)
		s = x_new - self.w
		return(s)
	    
		