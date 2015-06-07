import numpy as np
import scipy as sp
import random as rd
import itertools
import math
from collections import deque
import scipy.optimize

from stochastic_tools import sample_batch as chooseSample
from stochastic_tools import stochastic_gradient as calculateStochasticGradient

def hasTerminated(f, y, w, k, max_iter = 1e4):
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
	elif len(y) > 0 and np.linalg.norm(y[-1]) < eps:
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
	s = w-wPrevious
	#TODO: replace explicit stochastic gradient
	y = calculateStochasticGradient(g, w, X, z) - calculateStochasticGradient(g, wPrevious, X, z)

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
	(nSamples, nFeatures) = np.shape(X)
	
	if w1 == None:  w1 = np.zeros(dim)
	w = w1
	
	#Set wbar = wPrevious = 0
	wbar = np.zeros(w1.shape)
	gbar = np.zeros(w1.shape)
	wPrevious = np.zeros(w1.shape)
	
	# step sizes alpha_k
	alpha_k = beta
	alpha = lambda k: beta/(k + 1)

	s, y = deque(), deque()
	
	## accessed data points
	t = -1
	adp = 0
	
	for k in itertools.count():
		
		##
		## Check Termination Condition
		##
		if hasTerminated(f ,y ,w ,k, max_iter = max_iter):
			iterations = k
			break
		
		##
		## Draw mini batch
		##		
		X_S, z_S, adp = chooseSample(nSamples, X, z, b = batch_size, adp=adp)
		
		## 
		## Determine search direction
		##
		grad = calculateStochasticGradient(g, w, X_S, z_S)
		search_direction = -grad if k <= 2*L else -getH(s,y).dot(grad)
		
		##
		## Compute step size alpha
		##
		f_S = lambda x: f(x, X_S) if z == None else lambda x: f(x, X_S, z_S)
		g_S = lambda x: g(x, X_S) if z == None else lambda x: g(x, X_S, z_S)
		alpha_k = scipy.optimize.line_search(f_S, g_S, w, search_direction)[0]
		alpha_k = alpha(k) if alpha_k == None else alpha_k
		if debug: print "alpha", alpha_k
		
		##
		## Perform update
		##
		wPrevious = w
		w = w + alpha_k*search_direction
		wbar += w
		
		if debug: print "w: ", w
		
		##
		## compute Correction pairs every L iterations
		##
		if k%L == 0:
		    
			t += 1
			wbar /= float(L) 
			
			if t>0:
				#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH, adp = chooseSample(nSamples, X, z, b = batch_size_H, adp = adp)
				
				(s_t, y_t) = correctionPairs(g, w, wPrevious, X_SH, y_SH)
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft() 
					
			wbar = np.multiply(wbar, 0) 
			

	if iterations < max_iter:
	    print "Terminated successfully!" 
	print "Iterations:\t\t", iterations
	print "Accessed Data Points:\t", adp
	return w

