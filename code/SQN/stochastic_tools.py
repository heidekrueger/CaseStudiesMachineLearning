
'''
    Here we will list some tools which come in handy optimizing stochastic 
    functions
'''

import numpy as np
import random as rd
import math


def sample_batch(N, X, z = None, b = 1, debug = False):
	"""
	returns a subset of [N] as a list?

	Parameters:
		N: Size of the original set
		b: parameter for subsample size (e.g. b=.1)
	"""
	random_indices = rd.sample(range(N), int(math.ceil(b*N)) )
	if debug: print "random indices", random_indices
	if z == None or len(z) == 0:
		return np.asarray([X[i,:] for i in random_indices]), None
	else: 
		X_S = np.asarray([X[i,:] for i in random_indices])
		if debug: print z
		z_S = np.asarray([z[i] for i in random_indices])
		if debug: print X_S, z_S
		return X_S, z_S

def stochastic_gradient(g, w, X=None, z=None):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	(nSamples, nFeatures) = np.shape(X)
	if z is None:
		return sum([g(w,X[i,:]) for i in range(nSamples)])
	else:
		assert len(X)==len(z), "Error: Dimensions must match" 
		return sum([g(w,X[i,:],z[i]) for i in range(nSamples)])
 