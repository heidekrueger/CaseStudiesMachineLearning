import numpy as np
import scipy as sp
import random as rd
import itertools
import math
from collections import deque
import scipy.optimize

from stochastic_tools import sample_batch as chooseSample
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
	# H = (s_t^T y_t^T)/||y_t||^2 * I

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
	print w, wPrevious
	s = w - wPrevious
	#TODO: replace explicit stochastic gradient
	sg = lambda x: calculateStochasticGradient(g, x, X, z)
	y = ( sg(w) - sg(wPrevious) ) #/ ( np.linalg.norm(s) + 1)
	print "corr pair", s, y
	## 
	## Perlmutters Trick:
	## https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
	## H(x) v \approx \frac{g(x+r v) - g(x-r v)} {2r}
	## TODO: Not working??
	##r = 1e-2
	##y = ( sg(w + r*s) - sg(w - r*s) ) / 2*r
	
	return (s, y)


def solveSQN(f, g, X, z = None, w1 = None, dim = None, M=10, L=1.0, beta=1, batch_size = 1, batch_size_H = 1, max_iter = 1e4, debug = False):
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
	nSamples = len(z)
	nFeatures = len(X[0])
	
	if w1 == None:  
	    w1 = np.zeros(dim)[:,np.newaxis]
	w = w1
	
	#Set wbar = wPrevious = 0
	wbar = w1
	wPrevious = w
	print w.shape
	# step sizes alpha_k
	alpha_k = beta
	alpha = lambda k: beta/(k + 1)

	s, y = deque(), deque()
	
	## accessed data points
	t = -1
	adp = 0
	
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
		X_S, z_S, adp = chooseSample(X, z, b = batch_size, adp=adp)
		
		## 
		## Determine search direction
		##
		grad = calculateStochasticGradient(g, w, X_S, z_S)
		
		if True or k <= 2*L:
		    search_direction = -grad 
		else:
		    search_direction = -(getH(s,y).dot(grad))
		
		
		#search_direction = np.multiply( 1/np.linalg.norm(search_direction) , search_direction)
		
		if debug: print "Direction:", search_direction.T
		##
		## Compute step size alpha
		##
		#f_S = lambda x: f(x, X_S) if z == None else lambda x: f(x, X_S, z_S)
		#g_S = lambda x: g(x, X_S) if z == None else lambda x: g(x, X_S, z_S)
		
		f_S = lambda x: f(x, X_S, z_S)
		print "f", f_S(w)
		g_S = lambda x: calculateStochasticGradient(g, x, X_S, z_S)
		
		#alpha_k = scipy.optimize.line_search(f_S, g_S, w, search_direction)[0]
		#alpha_k = armijo_rule(f_S, g_S, w, search_direction, beta=.5, gamma= 1e-2 )
		alpha_k = 0.01   
		if debug: print "alpha", alpha_k
		
		##
		## Perform update
		##
		wPrevious = w
		
		print "LOL?!"
		print w
		#print search_direction
		#print alpha_k
		w = w + np.multiply(alpha_k, search_direction)
		wbar += w
		print ""
		if debug: print w
		print "" 
		##
		## compute Correction pairs every L iterations
		##
		if k%L == 0:
		    
			t += 1
			wbar /= float(L) 
			if t>0:
				#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH, adp = chooseSample(X, z, b = batch_size_H, adp = adp)
				
				(s_t, y_t) = correctionPairs(g, w, wPrevious, X_SH, y_SH)
				
				print "correction shapes", s_t, y_t
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft() 
					
			wbar = np.multiply(0, wbar) 
			

	if iterations < max_iter:
	    print "Terminated successfully!" 
	print "Iterations:\t\t", iterations
	print "Accessed Data Points:\t", adp
	return w

