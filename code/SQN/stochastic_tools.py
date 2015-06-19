
'''
    Here we will list some tools which come in handy optimizing stochastic 
    functions
'''

import numpy as np
import random as rd
import math


def sample_batch(w, X, z = None, b = None, r = None, debug = False):
	"""

	#TODO: Documentation Outdated!!!
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
	
	
	if debug: print X_S, z_S
		
	   
	if z == None or len(z) == 0:
		return X_S, None
	else: 
		return X_S, z_S

def stochastic_gradient(g, w, X=None, z=None):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	if X is not None:
		nSamples = len(X)
		nFeatures = len(X[0])
	#print nSamples
	#print X[0].shape, w.shape
	if X is None:
		return g(w)
	if z is None:
		return np.array(sum( [ g(w,X[i]) for i in range(nSamples) ] ))
	else:
		assert len(X)==len(z), "Error: Dimensions must match" 
		#print " one gradient:" , g(w,X[0],z[0])
		return sum([g(w,X[i],z[i]) for i in range(nSamples)])
 
def armijo_rule(f, g, x, s, start = 1.0, beta=.5, gamma= 1e-2 ):
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
	candidate = start
	#print "armijo"
	#print f(x + np.multiply(candidate, s)) 
	#print "fa", f(x)
	#print candidate * gamma * np.dot( g(x).T, s)
	#print s
	#print "---"
	while f(x + np.multiply(candidate, s)) - f(x) > candidate * gamma * np.dot( g(x).T, s) :
	
	#	print "armijo"
	#	print f(x + np.multiply(candidate, s)) - f(x)
	#	print candidate * gamma * np.dot( g(x).T, s)
		
		candidate *= beta
	return candidate


'''
    general tools
'''
def iter_to_array(iterator):
	return np.array([i for i in iterator])
    
def set_iter_values(iterator, w):
	for i in range(len(w)):
		iterator[i] = w[i]


