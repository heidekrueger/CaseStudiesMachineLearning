import numpy as np
import scipy as sp
import itertools
from collections import deque
import scipy.optimize


from stochastic_tools import sample_batch as chooseSample
from stochastic_tools import stochastic_gradient as calculateStochasticGradient
from stochastic_tools import armijo_rule
import SQN

def hasTerminated(f, wbars, gbars, k, max_iter = 1e4):
    if len(wbars) < 2 and len(gbars) < 2:
	return SQN.hasTerminated(f, [], [], k, max_iter)
    else:
	return SQN.hasTerminated(f, gbars[-1] - gbars[-2], wbars[-1] - wbars[-2], k, max_iter)

def get_correction_pairs(wbars, gbars):
	s, y = [], []
	for i in range(len(wbars)-1):
	    s.append(wbars[i+1] - wbars[i])
	    y.append(gbars[i+1] - gbars[i])
	return s, y

def getH(wbars, gbars):
    s, y = get_correction_pairs(wbars, gbars)
    return SQN.getH(s, y)
		
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
	gbar = w1
	wPrevious = w1

	# step sizes alpha_k
	alpha_k = beta
	alpha = lambda k: beta/(k + 1)

	wbars, gbars = deque(), deque()
		
	## accessed data points
	adp = 0
	
	for k in itertools.count():
		
		##
		## Check Termination Condition
		##
		if hasTerminated(f , wbars, gbars, k, max_iter = max_iter):
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
		if debug: print "f", f_S(w)
		g_S = lambda x: calculateStochasticGradient(g, x, X_S, z_S)
		
		#alpha_k = scipy.optimize.line_search(f_S, g_S, w, search_direction)[0]
		alpha_k = armijo_rule(f_S, g_S, w, search_direction, start = beta, beta=.5, gamma= 1e-2 )
		#alpha_k = 0.01   
		if debug: print "alpha", alpha_k
		
		##
		## Perform update
		##
		wPrevious = w
		wbar += w
		w = w + np.multiply(alpha_k, search_direction)
		gbar += grad
		
		if debug: print "w: ", w
				
		##
		## compute Correction pairs every L iterations
		##
		if k%L == 0:
			wbar = wbar / float(L)
			gbar = gbar / float(L)
			wbars.append(wbar)
			gbars.append(gbar)
			if len(wbars) > M + 1:
			    wbars.popleft()
			    gbars.popleft() 
			wbar = np.multiply(wbar, 0) 
			gbar = np.multiply(gbar, 0) 
	
	if iterations < max_iter:
	    print "Terminated successfully!" 
	print "Iterations:\t\t", iterations
	print "Accessed Data Points:\t", adp
	return w

