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



def hasTerminated(f, grad, w, k, max_iter = 1e4, debug=False):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		y: difference of gradients
		w: current variable
		k: current iteration

	"""
	if debug:
		print "Check termination"
		print "len grad:", np.linalg.norm(grad) 
		#print "fun val", f(w)
	eps = 1e-6
	if k > max_iter:
		return True
	#elif len(grad) > 0 and np.linalg.norm(grad) < eps:
	#	return True
	else:
		return False



def getH(s, y, debug = False):
	"""
	returns H_t as defined in algorithm 2
	"""
	
	assert len(s)>0, "s cannot be empty."
	assert len(s)==len(y), "s and y must have same length"
	
	assert s[0].shape == y[0].shape, "s and y must have same shape"
	assert abs(y[-1]).sum() != 0, "latest y entry cannot be 0!"
	assert 1/np.inner(y[-1], s[-1]) != 0, "!"
	# H = (s_t^T y_t^T)/||y_t||^2 * I


	# For now: Standard L-BFGS update
	# TODO: Two-Loop Recursion
	# TODO: Hardcode I each time to save memory. (Or sparse???)
	I= np.identity(len(s[0]))
	H = np.dot( (np.inner(s[-1], y[-1]) / np.inner(y[-1], y[-1])), I)
	for (s_j, y_j) in itertools.izip(s, y):
		rho = 1/np.inner(y_j, s_j)
		if debug: print s_j, y_j
		H = (I - rho* np.outer(s_j, y_j)).dot(H).dot(I - rho* np.outer(y_j, s_j))
		H += rho * np.outer(s_j, s_j) 
	return H

def correctionPairs(g, w, w_previous, X, z):
	"""
	returns correction pairs s,y

	"""
	s = w - w_previous
	#TODO: replace explicit stochastic gradient
	sg = lambda x: stochastic_gradient(g, x, X, z)
	y = ( sg(w) - sg(w_previous) ) #/ ( np.linalg.norm(s) + 1)
	# 
	# Perlmutters Trick:
	# https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
	# H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
	# TODO: Not working??
	#r = 1e-2
	#y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r
	
	return (s, y)

def solveSQN(f, g, X, z = None, w1 = None, dim = None, iterator = None, M=10, L=1.0, beta=1, batch_size = 1, batch_size_H = 1, max_iter = 1e4, debug = False, sampleFunction = None):
	"""
	Parameters:
		f:= f_i = f_i(omega, x, z[.]), loss function for one sample. The goal is to minimize
			F(omega,X,z) = 1/nSamples*sum_i(f(omega,X[i,:],z[i]))
			with respect to w
		g:= g_i = g_i(omega, x, z), gradient of f

		X: nSamples * nFeatures numpy array of Data
		z: nSamples * 1 numpy array of targets

		w1: initial w

		M: Memory-Parameter
	"""
	assert M > 0, "Memory Parameter M must be a positive integer!"
	assert w1 != None or dim != None or iterator != None, "Please privide either a starting point or the dimension of the optimization problem!"
	

	# dimensions
	nSamples = len(X)
	nFeatures = len(X[0])
	
	input_iterator = False
	if w1 is None and dim is None:  
	    input_iterator = True
	    w1 = stochastic_tools.iter_to_array(iterator)
	elif w1 is None:
	    w1 = np.zeros(dim)
	#    w1[0] = 3
	#    w1[0] = 4
	w = w1

	if sampleFunction != None:
		chooseSample = sampleFunction
	else:
		chooseSample = stochastic_tools.sample_batch

	#Set wbar = w_previous = 0
	wbar = w1
	w_previous = w
	if debug: print w.shape
	# step sizes alpha_k
	alpha_k = beta
	#alpha = lambda k: beta/(k + 1)

	s, y = deque(), deque()
	
	# accessed data points
	t = -1
	H = None
	for k in itertools.count():
		
		# Draw mini batch
		X_S, z_S= chooseSample(w=w, X=X, z=z, b = batch_size)
		if debug: print "sample:", chooseSample
		if len(X_S) == 0:
			X_S, z_S= chooseSample(w=w, X=X, z=z, b = batch_size)
		# Check Termination Condition
		if debug: print "Iteration", k
		if len(X_S) == 0 or hasTerminated(f , stochastic_gradient(g, w, X_S, z_S) ,w ,k, max_iter = max_iter, debug=True):
			iterations = k
			break
		
		# Determine search direction
		if k <= 2*L:  	search_direction = -stochastic_gradient(g, w, X_S, z_S)
		else:	   	search_direction = -H.dot(stochastic_gradient(g, w, X_S, z_S))
		if debug: 		print "Direction:", search_direction.T
	
		# Compute step size alpha
		f_S = lambda x: f(x, X_S, z_S) if z is not None else f(x, X_S)
		g_S = lambda x: stochastic_gradient(g, x, X_S, z_S)
		alpha_k = armijo_rule(f_S, g_S, w, search_direction, start = beta, beta=.5, gamma= 1e-2 )
		alpha_k = max([alpha_k, 1e-5])
		    
		if debug: print "f\n", f_S(w)
		if debug: print "w\n", w
		if debug: print "alpha", alpha_k
		
		# Perform update
		w_previous = w
		w = w + np.multiply(alpha_k, search_direction)
		wbar += w
		# compute Correction pairs every L iterations
		if k%L == 0:
			t += 1
			wbar /= float(L) 
			if t>0:
				#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH = chooseSample(w, X, z, b = batch_size_H)
				
				(s_t, y_t) = correctionPairs(g, w, w_previous, X_SH, y_SH)
				
				if debug: print "correction shapes", s_t, y_t
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft()
				
				H = getH(s, y)
				
			wbar = np.zeros(dim)

	if iterations < max_iter:
		print "Terminated successfully!" 
	print "Iterations:\t\t", iterations
	
	if input_iterator:  
		stochastic_tools.set_iter_values(iterator, w)
		return iterator
	else:
		return w
		



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
    
    
class SQN:
	"""
	TODO: Two-Loop-Recursion!
	"""
	
	def __init__(self):
		"""
		for initialization: Provide one of w1, dim or flat iterator object
		M: Memory-Parameter
		L: Compute Hessian information every Lth step
		beta: start configuration for line search
		batch_size: number of samples to be drawn for gradient evaluation
		batch_size: number of samples to be drawn for Hessian approximation
		max_iter: Terminate after this many steps
		debug: Print progress statements
		"""
		self.options = dict()
		self.options['w1'] =  None
		self.options['dim'] =  None
		self.options['iterator'] =  None
		self.options['sampleFunction'] =  stochastic_tools.sample_batch
		self.options['M'] =  10
		self.options['L'] =  1
		self.options['beta'] =  1
		self.options['batch_size'] =  1
		self.options['batch_size_H'] =  1
		self.options['max_iter'] =  1e3
		self.debug =  False
		
		self.s, self.y = deque(), deque()
		self.w, self.w_previous = None, None
		self.wbar = None
		self.wbar_previous = None
		
		self.termination_counter = 0
		self.iterations = 0
		self.iterator = None
		
	def set_options(self, options):
		for key in options:
			if key in self.options:
				self.options[key] = options[key]
				
	def get_options(self):
		return self.options
	   
	def print_options(self):
		for key in self.options:
			print key
			print self.options[key]
	
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
		    w1 = np.zeros(dim)
		
		self.options['dim'] = len(w1)
		    
		# init
		self.w = w1
		self.w_previous = self.w
		self.wbar = np.zeros(self.w.shape)
		self.wbar_previous = None
		self.s, self.y = deque(), deque()
		if self.debug: print self.w.shape
		return
	    
	def has_terminated(self, grad, w):
		"""
		Checks whether the algorithm has terminated

		Parameters:
			grad: gradient
			w: current variable
		"""
		return hasTerminated(id, grad, w, 0, debug=self.debug)
		
	def get_H(self, debug = False):
		"""
		returns H_t as defined in algorithm 2
		TODO: Two-Loop-Recursion
		"""
		return getH(self.s, self.y, debug)
	    
	def solve(self, f, g, X, z = None):
		"""
		Parameters:
			f:= f_i = f_i(omega, x, z[.]), loss function for one sample. The goal is to minimize
				F(omega,X,z) = 1/nSamples*sum_i(f(omega,X[i,:],z[i]))
				with respect to w
			g:= g_i = g_i(omega, x, z), gradient of f

			X: list of nFeatures numpy column arrays of Data
			z: list of nSamples integer labels
		"""
		
		self.set_start(w1=self.options['w1'], dim=self.options['dim'], iterator=self.options['iterator'])
		
		for k in itertools.count():
		    
			if self.debug: print "Iteration", k
			
			self.w = self.solve_step(f, g, X, z, k)
			
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
		
	def solve_step(self, f, g, X, z, k):
		"""
		perform one update step
		"""
		assert self.w is not None, "Error! weights not initialized!"
		# perform gradient update using armijo rule and hessian information
		self.w = self.gradient_step(f, g, X, z)
		if self.debug: print self.w
		# update wbar and get new correction pairs
		self.wbar += self.w
		if k % self.options['L'] == 0:
			self.wbar /= float(self.options['L']) 
			if self.wbar_previous is not None:
				# draw hessian sample and get the corresponding stochastic gradient
				X_SH, y_SH = self.options['sampleFunction'](self.w, X, z, b = self.options['batch_size_H'])
				g_SH = lambda x: stochastic_gradient(g, x, X_SH, y_SH)
				# do update
				self.update_correction_pairs(g_SH, self.wbar, self.wbar_previous)
			# save old mean location
			self.wbar_previous = self.wbar
			self.wbar = np.zeros(self.options['dim'])
		return self.w
		
	def gradient_step(self, f, g, X, z):
		"""
		do the gradient updating rule
		"""
		# Draw mini batch
		X_S, z_S= self.options['sampleFunction'](w=self.w, X=X, z=z, b = self.options['batch_size'])
		# Stochastic functions
		f_S = lambda x: f(x, X_S, z_S) if z is not None else f(x, X_S)
		g_S = lambda x: stochastic_gradient(g, x, X_S, z_S)
		# Determine search direction
		if len(self.y) < 2:
			search_direction = -g_S(self.w)
		else:
			H = self.get_H()
			search_direction = -H.dot(g_S(self.w))
		if self.debug: print "Direction:", search_direction.T
		# Compute step size alpha
		alpha = armijo_rule(f_S, g_S, self.w, search_direction, start = self.options['beta'], beta=.5, gamma= 1e-2 )
		alpha = max([alpha, 1e-5])
		if self.debug: print "step size: ", alpha
		# Check Termination Condition
		if len(X_S) == 0 or self.has_terminated(g_S(self.w) , self.w):
			self.termination_counter += 1
			return self.w
		# Update
		self.w = self.w + np.multiply(alpha, search_direction)
		return self.w
	    
	def update_correction_pairs(self, g_S, w, w_previous):
		    """
		    returns correction pairs s,y
		    TODO: replace explicit stochastic gradient
		    
		    Perlmutters Trick:
		    https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
		    H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
		    r = 1e-2
		    y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r
		    """
		    r = 0.01
		    s_t = w - w_previous
		    s_t = np.multiply(r, s_t)
		    y_t = (g_S(w) - g_S(w - s_t)) / (r)
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
