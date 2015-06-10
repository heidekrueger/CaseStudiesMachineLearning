import numpy as np
import scipy as sp
import random as rd
import itertools
import math
from collections import deque
import scipy.optimize

import stochastic_tools
from stochastic_tools import stochastic_gradient as calculateStochasticGradient
from stochastic_tools import armijo_rule

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

def correctionPairs(g, w, wPrevious, X, z):
	"""
	returns correction pairs s,y

	"""
	s = w - wPrevious
	#TODO: replace explicit stochastic gradient
	sg = lambda x: calculateStochasticGradient(g, x, X, z)
	y = ( sg(w) - sg(wPrevious) ) #/ ( np.linalg.norm(s) + 1)
	## 
	## Perlmutters Trick:
	## https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
	## H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
	## TODO: Not working??
	##r = 1e-2
	##y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r
	
	return (s, y)


def solveSQN(f, g, X, z = None, w1 = None, dim = None, M=10, L=1.0, beta=1, batch_size = 1, batch_size_H = 1, max_iter = 1e4, debug = False, sampleFunction = None):
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
	assert w1 != None or dim != None, "Please privide either a starting point or the dimension of the optimization problem!"
	

	## dimensions
	nSamples = len(X)
	nFeatures = len(X[0])
	
	if w1 == None:  
	    w1 = np.zeros(dim)
	#    w1[0] = 3
	#    w1[0] = 4
	w = w1

	if sampleFunction != None:
		chooseSample = sampleFunction
	else:
		chooseSample = stochastic_tools.sample_batch

	#Set wbar = wPrevious = 0
	wbar = w1
	wPrevious = w
	if debug: print w.shape
	# step sizes alpha_k
	alpha_k = beta
	#alpha = lambda k: beta/(k + 1)

	s, y = deque(), deque()
	
	## accessed data points
	t = -1
	
	for k in itertools.count():
		
		if debug: print "Iteration", k
		##
		## Check Termination Condition
		##
		if hasTerminated(f , calculateStochasticGradient(g, w, X, z) ,w ,k, max_iter = max_iter):
			iterations = k
			break
		
		##
		## Draw mini batch
		##		
		if debug:
			print "debug sample function:", chooseSample
		X_S, z_S= chooseSample(w=w, X=X, z=z, b = batch_size)
		
		## 
		## Determine search direction
		##
		grad = calculateStochasticGradient(g, w, X_S, z_S)
		
		if k <= 2*L:
		    search_direction = -grad 
		else:
		    search_direction = -(getH(s,y).dot(grad))
		
		if debug: print "Direction:", search_direction.T
	
		##
		## Compute step size alpha
		##
		if z is None:
		    f_S = lambda x: f(x, X_S)
		    g_S = lambda x: calculateStochasticGradient(g, x, X_S)
		else:
		    f_S = lambda x: f(x, X_S, z_S)
		    g_S = lambda x: calculateStochasticGradient(g, x, X_S, z_S)

		alpha_k = armijo_rule(f_S, g_S, w, search_direction, start = beta, beta=.5, gamma= 1e-2 )
		if alpha_k < 1e-5:
		    alpha_k = 1e-5
		    
		if debug: print "f\n", f_S(w)
		if debug: print "w\n", w
		if debug: print "alpha", alpha_k
		
		##
		## Perform update
		##
		wPrevious = w
		w = w + np.multiply(alpha_k, search_direction)
		wbar += w
		
		##
		## compute Correction pairs every L iterations
		##
		if k%L == 0:
		    
			t += 1
			wbar /= float(L) 
			if t>0:
				#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH = chooseSample(w, X, z, b = batch_size_H)
				
				(s_t, y_t) = correctionPairs(g, w, wPrevious, X_SH, y_SH)
				
				if debug: print "correction shapes", s_t, y_t
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft() 
					
			wbar = np.zeros(dim)

	if iterations < max_iter:
		print "Terminated successfully!" 
	# print "Iterations:\t\t", iterations
	return w
