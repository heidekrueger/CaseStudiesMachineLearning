
'''
    Here we will list some tools which come in handy optimizing stochastic 
    functions
'''

import numpy as np
import random as rd
import math


def sample_batch(X, z = None, b = None, r = None, debug = False, adp = None):
	"""
	returns a subset of [N] as a list?

	Parameters:
		N: Size of the original set
		b: parameter for subsample size (e.g. b=.1)
	"""
	
	assert b != None or r!= None, "Choose either absolute or relative sample size!"
	assert (b != None) != (r!= None), "Choose only one: Absolute or relative sample size!"
	N = len(X)
	if b != None:
	    nSamples = b
	else:
	    nSamples = r*N
	if nSamples > N:
	    if debug:
		print "Batch size larger than N, using whole dataset"
	    nSamples = N
	##
	## Draw from uniform distribution
	##
	random_indices = rd.sample( range(N), int(nSamples)) 
	if debug: print "random indices", random_indices
	 
	X_S = np.asarray([X[i] for i in random_indices])
	z_S = np.asarray([z[i] for i in random_indices]) if z != None else None
	
	##
	## Count data points
	##
	if adp != None and type(adp) == type(1):
	    adp += nSamples
	
	if debug: print X_S, z_S, adp
		
	   
	if z == None or len(z) == 0:
		return X_S, None, adp
	else: 
		return X_S, z_S, adp

def stochastic_gradient(g, w, X=None, z=None):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	nSamples = len(z)
	nFeatures = len(X[0])
	#print nSamples
	#print X[0].shape, w.shape
	if z is None:
		return np.matrix(sum( [ g(w,X[i]) for i in range(nSamples) ] ))
	else:
		assert len(X)==len(z), "Error: Dimensions must match" 
		#print " one gradient:" , g(w,X[0],z[0])
		return sum([g(w,X[i],z[i]) for i in range(nSamples)])
 
def armijo_rule(f, g, x, s, beta=.5, gamma= 1e-2 ):
	"""
	Determines the armijo-rule step size alpha for approximating 
	line search min f(x+omega*s)

	Parameters:
		f: objective function
		g: gradient
		x:= x_k
		s:= x_k search direction
		beta, gamma: parameters of rule
	"""
	candidate = 10
	#print "armijo"
	#print f(x + np.multiply(candidate, s)) 
	#print "fa", f(x)
	#print candidate * gamma * np.dot( g(x).T, s)
	#print s
	#print "---"
	while f(x + np.multiply(candidate, s)) < 1e4 and f(x + np.multiply(candidate, s)) - f(x) > candidate * gamma * np.dot( g(x).T, s) :
	#	print "armijo"
	#	print f(x + np.multiply(candidate, s)) - f(x)
	#	print candidate * gamma * np.dot( g(x).T, s)
		
		candidate *= beta
	return candidate





