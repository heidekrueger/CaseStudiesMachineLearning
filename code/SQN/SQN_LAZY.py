import numpy as np
import scipy as sp
import itertools
from collections import deque
import scipy.optimize


from stochastic_tools import sample_batch as chooseSample
from stochastic_tools import stochastic_gradient as calculateStochasticGradient


def hasTerminated(f, wbars, gbars, k, max_iter = 1e4):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		g: gradient entry for one sample
		w: current variable
		k: current iteration

	"""
	eps = 1e-6
	if k > max_iter:
		return True
	elif len(gbars) > 1 and np.linalg.norm(gbars[-1] - gbars[-2]) < eps:
		return True
	else:
		return False


def getH(wbars, gbars):
	"""
	returns H_t as defined in algorithm 2
	"""
	
	assert len(wbars)>0, "s cannot be empty."
	assert len(wbars)==len(gbars), "s and y must have same length"
	assert wbars[0].shape == gbars[0].shape, "s and y must have same shape"
	
	##
	## get correction pairs
	##
	s, y = [], []
	for i in range(len(wbars)-1):
	    s.append(wbars[i+1] - wbars[i])
	    y.append((gbars[i+1]) - (gbars[i]))
	#s = [ np.array(a) + np.array(b) for (a,b) in zip( list(wbars)[1:] - list(wbars)[:-1] ) ]
	#y = [ a + b for (a,b) in zip( list(gbars)[1:] - list(gbars)[:-1] ) ]
	assert abs(y[-1]).sum() != 0, "latest y entry cannot be 0!"
	
	# H = (s_t^T y_t^T)/||y_t||^2 * I
	# TODO: Two-Loop Recursion
	# TODO: Hardcode I each time to save memory. (Or sparse???)
	I= np.identity(len(s[0]))
	H = np.dot( (np.inner(s[-1], y[-1]) / np.inner(y[-1], y[-1])), I)
	for (s_j, y_j) in itertools.izip(s, y):
		rho = 1/np.inner(y_j, s_j)
		H = (I - rho* np.outer(s_j, y_j)).dot(H).dot(I - rho* np.outer(y_j, s_j))
		H += rho * np.outer(s_j, s_j) 

	return H

		
def solveSQN(f, g, X, z = None, w1 = None, M=10, L=1.0, beta=1, batch_size = 1, debug = False):
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

	##
	## TODO: Give dimensions!
	##
	if w1 == None:
		w1 = np.zeros(2)
	w = w1
	
	## dimensions
	(nSamples, nFeatures) = np.shape(X)
	
	#Set wbar = wPrevious = 0
	wbar = np.zeros(w1.shape)
	gbar = np.zeros(w1.shape)
	wPrevious = wbar

	# step sizes alpha_k
	alpha_k = beta
	alpha = lambda k: beta/(k + 1)

	wbars, gbars = deque(), deque()

	adp = 0
		
	alpha_counter = 0
	for k in itertools.count():
		if hasTerminated(f , wbars, gbars, k):
			if debug: print "terminated"
			iterations = k
			break
		
		##
		## Draw mini batch
		##		
		X_S, z_S = chooseSample(nSamples, X, z, b = batch_size)
		adp += batch_size
		
		## 
		## Determine search direction
		##
		grad = calculateStochasticGradient(g, w, X_S, z_S)
		search_dir = -grad if k <= 2*L else -getH(wbars,gbars).dot(grad)
		
		##
		## Compute step size alpha
		##
		f_S = lambda x: f(x, X_S) if z == None else lambda x: f(x, X_S, z_S)
		g_S = lambda x: g(x, X_S) if z == None else lambda x: g(x, X_S, z_S)
		alpha_k = scipy.optimize.line_search(f_S, g_S, w, search_dir)[0]
		alpha_k = alpha(k) if alpha_k == None else alpha_k
		if debug: print "alpha", alpha_k
		
		##
		## Perform update
		##
		wbarPrevious = wbar
		wPrevious = w
		w = w + alpha_k*search_dir
		if debug: print "w: ", w
		
		wbar += w
		gbar += grad
				
		##
		## compute Correction pairs every L iterations
		##
		if k%L == 0:
			wbar = wbar/float(L)
			gbar = gbar/float(L)
			wbars.append(wbar)
			gbars.append(gbar)
			if len(wbars) > M + 1:
			    wbars.popleft()
			    gbars.popleft() 	
			wbar = 0
			gbar *= 0
			
	print iterations, alpha_counter, adp
	return w

