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



def hasTerminated(f, grad, w, k, max_iter = 1e4):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		y: difference of gradients
		w: current variable
		k: current iteration

	"""
	eps = 1e-6
	if k > max_iter:
		return True
	elif len(grad) > 0 and np.linalg.norm(grad) < eps:
		return True
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
		if len(X_S) == 0 or hasTerminated(f , stochastic_gradient(g, w, X_S, z_S) ,w ,k, max_iter = max_iter):
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
	
	def __init__(self):
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
		self.options['debug'] =  True
		
		
	def set_options(self, options):
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
		for key in options:
			if key in self.options:
				self.options[key] = options[key]
				
	
	def gradient_step(self, f, g, X, z, w, H, k):
		
		# Draw mini batch
		X_S, z_S= self.options['sampleFunction'](w=w, X=X, z=z, b = self.options['batch_size'])
		if len(X_S) == 0:
			X_S, z_S= chooseSample(w=w, X=X, z=z, b = self.options['batch_size'])
		
		# Stochastic functions
		f_S = lambda x: f(x, X_S, z_S) if z is not None else f(x, X_S)
		g_S = lambda x: stochastic_gradient(g, x, X_S, z_S)

		
		# Check Termination Condition
		if self.options['debug']: print "Iteration", k
		if len(X_S) == 0 or hasTerminated(f , g_S(w) , w ,k, max_iter = self.options['max_iter']):
			return w, True
			
		# Determine search direction
		if k <= 2*self.options['L']:  	search_direction = -stochastic_gradient(g, w, X_S, z_S)
		else:	   				search_direction = -H.dot(stochastic_gradient(g, w, X_S, z_S))
		if self.options['debug']: 		print "Direction:", search_direction.T

		# Compute step size alpha
		alpha_k = armijo_rule(f_S, g_S, w, search_direction, start = self.options['beta'], beta=.5, gamma= 1e-2 )
		alpha_k = max([alpha_k, 1e-5])
		
		# Update
		w = w + np.multiply(alpha_k, search_direction)
		return w, False
	    
	def Hessian_step(self, g, X, z, w, w_previous, s, y):
		#choose a Sample S_H \subset [nSamples] to define Hbar
		X_SH, y_SH = self.options['sampleFunction'](w, X, z, b = self.options['batch_size_H'])
		(s_t, y_t) = correctionPairs(g, w, w_previous, X_SH, y_SH)
		
		s.append(s_t)
		y.append(y_t)
		
		if len(s) > self.options['M']:
			s.popleft()
			y.popleft()
			
		if self.options['debug']: print len(s), len(y)

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
		
		assert self.options['M'] > 0, "Memory Parameter M must be a positive integer!"
		
		# dimensions
		nSamples = len(X)
		nFeatures = len(X[0])
		
		# start point
		assert self.options['w1'] != None or self.options['dim'] != None or self.options['iterator'] != None, \
		    "Please privide either a starting point or the dimension of the optimization problem!"
		if self.options['w1'] is None and self.options['dim'] is None:  
		    w1 = stochastic_tools.iter_to_array(self.options['iterator'])
		elif self.options['w1'] is None:
		    w1 = np.zeros(self.options['dim'])
		else:
		    w1 = self.options['w1']
		
		self.options['dim'] = len(w1)
		    
		# init
		w = w1
		w_previous = w
		wbar = np.zeros(w.shape)
		s, y = deque(), deque()
		alpha_k = self.options['beta']
		H = None
		
		if self.options['debug']: print w.shape
		
		t = -1
		for k in itertools.count():
			
		    w_previous = w
		    w, terminated = self.gradient_step(f, g, X, z, w, H, k)
		    if self.options['debug']: print w
		    
		    if terminated:
			    iterations = k
			    break
		    
		    wbar += w
		    if k % self.options['L'] == 0:
			    t += 1
			    wbar /= float(self.options['L']) 
			    if t>0:
				    self.Hessian_step(g, X, z, w, w_previous, s, y)
				    H = getH(s, y)
			    wbar = np.zeros(self.options['dim'])
		
		if iterations < self.options['max_iter']:
			print "Terminated successfully!" 
		print "Iterations:\t\t", iterations
		
		if self.options['iterator'] is not None:  
			stochastic_tools.set_iter_values(self.options['iterator'], w)
			return iterator
		else:
			return w
		    