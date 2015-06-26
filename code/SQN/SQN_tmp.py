import numpy as np
import scipy as sp
import random as rd
import itertools
import math
from collections import deque
import scipy.optimize

import stochastic_tools
from stochastic_tools import stochastic_gradient
from stochastic_tools import armijo_rule


"""
TODO: Iterator support not yet tested! Try on Dictionary Learning Problem!
"""

def solveSQN(f, g, X, z = None, w1 = None, dim = None, iterator = None, M=10, L=1.0, beta=1, batch_size = 1, batch_size_H = 1, max_iter = 1e4, debug = False, sampleFunction = stochastic_tools.sample_batch):
	""" 
	function wrapper for SQN class. Used for backward compatibility.
	"""
	sqn = SQN()
	sqn.debug = debug
	sqn.set_options({'w1':w1, 'dim':dim, 'iterator':iterator, 'M':M, 'L':L, 'beta':beta, 'batch_size': batch_size, 'batch_size_H': batch_size_H, 'max_iter': max_iter, 'sampleFunction': sampleFunction})
	return sqn.solve(f, g, X, z)

class Optimizer():
	"""
	for initialization: Provide one of w1, dim or flat iterator object
	max_iter: Terminate after this many steps
	debug: Print progress statements
	"""
	def __init__(self):
		self.options = dict()
		self.options['w1'] =  None
		self.options['dim'] =  None
		self.options['iterator'] =  None
		self.options['max_iter'] =  1e3
		
		self.debug =  False
		
		self.termination_counter = 0
		self.iterations = 0
		self.iterator = None
	
		self.w, self.w_previous = None, None
		self.f_vals = []
		self.gradients = []
	
		
	def set_options(self, options):
		for key in options:
			if key in self.options:
				self.options[key] = options[key]

	def set_option(self, key, value):
		if key in options:
			self.options[key] = value
				
	def get_options(self):
		return self.options
	   
	def print_options(self):
		for key in self.options:
			print key
			print self.options[key]
	
	def get_position(self):
		return self.w
	
	def set_start(self, w1=None, dim=None, iterator=None):
		"""
		Set start point of the optimization using numpy array, dim or flat.iterator object.
		"""
		assert self.options['M'] > 0, "Memory Parameter M must be a positive integer!"
		# start point
		assert w1 is not None or dim is not None or iterator is not None, \
		    "Please privide either a starting point or the dimension of the optimization problem!"

		if w1 is None and dim is None:  
		    self.options['iterator'] = iterator
		    w1 = stochastic_tools.iter_to_array(self.options['iterator'])
		elif w1 is None:
		    w1 = np.ones(dim)
		
		self.options['dim'] = len(w1)
		
		self.w = w1
		return
	
	    
	def solve(self, f, g, X = None, z = None):
		"""
		Parameters:
			f:= f_i = f_i(omega, x, z[.]), loss function for one sample. The goal is to minimize
				F(omega,X,z) = 1/nSamples*sum_i(f(omega,X[i,:],z[i]))
				with respect to w
			g:= g_i = g_i(omega, x, z), gradient of f

			X: list of nFeatures numpy column arrays of Data
			z: list of nSamples integer labels
		"""
		
		assert X is not None or self.options['sampleFunction'] is not None, \
			"Please provide either a data set or a sampling function"
		
		self.set_start(w1=self.options['w1'], dim=self.options['dim'], iterator=self.options['iterator'])
		
		for k in itertools.count():
		    
			if self.debug: print "Iteration", k
			
			self.w = self.solve_one_step(f, g, X, z, k)
			
			if k > self.options['max_iter'] or self.termination_counter > 4:
			    self.iterations = k
			    break
			
		if self.iterations < self.options['max_iter']:
			print "Terminated successfully!" 
		print "Iterations:\t\t", self.iterations
		
		if self.options['iterator'] is not None:  
			stochastic_tools.set_iter_values(self.options['iterator'], self.w)
			return iterator
		else:
			return self.w
		
	def solve_one_step(self, f, g, X=None, z=None, k=1):
		"""
		perform one update step
		"""
		assert self.w is not None, "Error! weights not initialized!"
		raise NotImplementedError
		# perform gradient update using armijo rule and hessian information
		self.w = self._perform_update(f, g, X, z)
		if self.debug: print self.w
		
		# update wbar and get new correction pairs
		self.wbar += self.w
		if k % self.options['L'] == 0:
			self.wbar /= float(self.options['L']) 
			if self.wbar_previous is not None:
				if self.debug: print "HESSE"
				self._update_correction_pairs(g, X, z)
			self.wbar_previous = self.wbar
			self.wbar = np.zeros(self.options['dim'])
		return self.w
	
	# Determine search direction
	def _get_search_direction(self, g_S):
		raise NotImplementedError
		return search_direction
	    
	# Calculate gradient and perform update
	def _perform_update(self, f, g, X, z):
		raise NotImplementedError
		return self.w
	
	def _has_terminated(self, grad, w):
		"""
		Checks whether the algorithm has terminated
		Parameters:
			grad: gradient
			w: current variable
		"""
		if self.debug:
			print "Check termination"
			print "len grad:", np.linalg.norm(grad) 
		eps = 1e-6
		if len(grad) > 0 and np.linalg.norm(grad) < eps:
			return True
		else:
			return False

class StochasticOptimizer(Optimizer):
	"""
	for initialization: Provide one of w1, dim or flat iterator object
	M: Memory-Parameter
	L: Compute Hessian information every Lth step
	beta: start configuration for line search
	batch_size: number of samples to be drawn for gradient evaluation
	batch_size: number of samples to be drawn for Hessian approximation
	"""
	def __init__(self):
		Optimizer.__init__(self)
		self.options['sampleFunction'] =  stochastic_tools.sample_batch
		self.options['M'] =  10
		self.options['L'] =  1
		self.options['N'] =  None
		self.options['beta'] =  1
		self.options['batch_size'] =  1
		self.options['batch_size_H'] =  1
		
	def _draw_sample(self, X, z=None, b=None, recursion_depth = 1):
		"""
		Draw sample from smaple function. Recurse if empty sample was drawn.
		"""
		if b is None: b = self.options['batch_size']
		if X is None and self.options['N'] is None:
			X_S, z_S= self.options['sampleFunction'](self.w, self.options['N'], b = b)
		elif X is None and self.options['N'] is not None:
			X_S, z_S= self.options['sampleFunction'](self.w, self.options['N'], b = b)
		else: 
			X_S, z_S= self.options['sampleFunction'](self.w, X, z=z, b =b )
		
		# if empty sample try one more time:
		if len(X_S) == 0 and recursion_depth > 0:
			X_S, z_S = self._draw_sample(X, z=z, b=b, recursion_depth=recursion_depth-1)
		if self.debug:
			print "sample length:", len(X_S), len(z_S)
		return X_S, z_S
	    
class SQN(StochasticOptimizer):
	"""
	TODO: Two-Loop-Recursion!
	"""

	def __init__(self):
		self.s, self.y = deque(), deque()
		self.w, self.w_previous = None, None
		self.wbar = None
		self.wbar_previous = None
		self.f_vals = []
		self.gradients = []
		StochasticOptimizer.__init__(self)
		#super(SQN, self).__init__()
	    
	def solve_one_step(self, f, g, X=None, z=None, k=1):
		"""
		perform one update step
		"""
		assert self.w is not None, "Error! weights not initialized!"
		
		# perform gradient update using armijo rule and hessian information
		self.w = self._perform_update(f, g, X, z)
		if self.debug: print self.w
		
		# update wbar and get new correction pairs
		self.wbar += self.w
		if k % self.options['L'] == 0:
			self.wbar /= float(self.options['L']) 
			if self.wbar_previous is not None:
				if self.debug: print "HESSE"
				self._update_correction_pairs(g, X, z)
			self.wbar_previous = self.wbar
			self.wbar = np.zeros(self.options['dim'])
		return self.w
	
	# Determine search direction
	def _get_search_direction(self, g_S):
		if len(self.y) < 2:
			search_direction = -g_S(self.w)
		else:
			#search_direction = -self._two_loop_recursion(g_S)
			H = self.get_H()
			search_direction = -H.dot(g_S(self.w))
		if self.debug: print "Direction:", search_direction.T
		return search_direction
	    
	# Calculate gradient and perform update
	def _perform_update(self, f, g, X, z):
		"""
		do the gradient updating rule
		"""
		# Draw sample batch
		if X is None:
			X_S, z_S = self._draw_sample(self.options['N'], b = self.options['batch_size'])
		else:
			X_S, z_S = self._draw_sample(X, z, b=self.options['batch_size'])
		# Stochastic functions
		f_S = lambda x: f(x, X_S, z_S) #if z is not None else f(x, X_S)
		g_S = lambda x: stochastic_gradient(g, x, X_S, z_S)
		
		# Get search direction
		search_direction = self._get_search_direction(g_S)
		
		# Line Search
		alpha = armijo_rule(f_S, g_S, self.w, search_direction, start = self.options['beta'], beta=.5, gamma= 1e-2 )
		alpha = max([alpha, 1e-5])
		if self.debug: print "step size: ", alpha
		
		# Check Termination Condition
		if len(X_S) == 0 or self._has_terminated(g_S(self.w) , self.w):
			self.termination_counter += 1
			return self.w
		
		# Update
		self.w = self.w + np.multiply(alpha, search_direction)
		
		self.f_vals.append( f_S(self.w) )
		self.gradients.append( g_S(self.w) )

		return self.w
	    
	def _update_correction_pairs(self, g, X, z):
		"""
		returns correction pairs s,y
		TODO: replace explicit stochastic gradient
		
		Perlmutters Trick:
		https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
		H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
		r = 1e-2
		y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r
		"""
		
		# draw hessian sample and get the corresponding stochastic gradient
		X_SH, y_SH = self._draw_sample(X, z, b = self.options['batch_size_H'])
		g_SH = lambda x: stochastic_gradient(g, x, X_SH, y_SH)
			    
		r = 0.01
		s_t = self.wbar - self.wbar_previous
		s_t = np.multiply(r, s_t)
		y_t = (g_SH(self.wbar) - g_SH(self.wbar - s_t)) / (r)
		
		if self.debug:
			print "correction:"
			print "s_t: ", s_t
			print "y_t: ", y_t
			
		if abs(y_t).sum() != 0:
		    self.s.append(s_t)
		    self.y.append(y_t)
		else:
		    print "PROBLEM! zero y"
		    
		if len(self.s) > self.options['M']:
			self.s.popleft()
			self.y.popleft()
			
		if self.debug: print "Length s, y:", len(self.s), len(self.y)
		return
	
	

	def get_H(self, debug = False):
		"""
		returns H_t as defined in algorithm 2
		TODO: Two-Loop-Recursion
		"""
		assert len(self.s)>0, "s cannot be empty."
		assert len(self.s)==len(self.y), "s and y must have same length"
		
		assert self.s[0].shape == self.y[0].shape, "s and y must have same shape"
		assert abs(self.y[-1]).sum() != 0, "latest y entry cannot be 0!"
		assert 1/np.inner(self.y[-1], self.s[-1]) != 0, "!"
		# H = (s_t^T y_t^T)/||y_t||^2 * I

		# For now: Standard L-BFGS update
		# TODO: Two-Loop Recursion
		# TODO: Hardcode I each time to save memory. (Or sparse???)
		I= np.identity(len(self.s[0]))
		H = np.dot( (np.inner(self.s[-1], self.y[-1]) / np.inner(self.y[-1], self.y[-1])), I)
		for (s_j, y_j) in itertools.izip(self.s, self.y):
			rho = 1/np.inner(y_j, s_j)
			if debug: print s_j, y_j
			H = (I - rho* np.outer(s_j, y_j)).dot(H).dot(I - rho* np.outer(y_j, s_j))
			H += rho * np.outer(s_j, s_j) 
		return H

	def _two_loop_recursion(self, g_S):
		"""
		TODO: Description two loop recursion and wikipedia link
		TODO: Check and TEST!!
		returns:
		z = H_k g_k
		"""
	    
		assert len(s)>0, "s cannot be empty."
		assert len(s)==len(y), "s and y must have same length"
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
			q = q - np.multiply( a[j] - b_j, s[j])
		
		return z
		
