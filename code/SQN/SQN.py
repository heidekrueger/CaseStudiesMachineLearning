import numpy as np
import scipy as sp
import random as rd
import itertools
import math
from collections import deque

def hasTerminated(f,y,w,k, max_iter = 100):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		g: gradient entry for one sample
		w: current variable
		k: current iteration

	"""
	
	eps = 1e-4
	if k > max_iter:
		return True
	elif len(y) > 0 and np.linalg.norm(y[-1]) < eps:
		return True
	else:
		return False


def chooseSample(N, X, z = None, b=1):
	"""
	returns a subset of [N] as a list?

	Parameters:
		N: Size of the original set
		b: parameter for subsample size (e.g. b=.1)
	"""
	random_indices = rd.sample(range(N), int(math.ceil(b*N)) )
	print "random indices", random_indices
	if z == None or len(z) == 0:
		return np.asarray([X[i,:] for i in random_indices]), None
	else: 
		X_S = np.asarray([X[i,:] for i in random_indices])
		print z
		z_S = np.asarray([z[i] for i in random_indices])
		print X_S, z_S
		return X_S, z_S

def calculateStochasticGradient(w, g, X, z=None):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	### TODO: FEHLER!!! WENN KEINE SAMPLES DANN FALSCHE WERTE????
	print np.shape(X)
	(nSamples, nFeatures) = np.shape(X)
	if z is None:
		return sum([g(w,X[i,:]) for i in range(np.shape(X)[0])])
	else:
		assert len(X)==len(z), "Error: Dimensions mus match" 
		return sum([g(w,X[i,:],z[i]) for i in range(X.shape[0])])


def getH(s, y):
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
		print s_j, y_j
		H = (I - rho* np.outer(s_j, y_j)).dot(H).dot(I - rho* np.outer(y_j, s_j))
		H += rho * np.outer(s_j, s_j) 

	return H

def correctionPairs(w, wPrevious, g, X, z):
	"""
	returns correction pairs s,y

	"""
	s = w-wPrevious
	print "s", s
	#TODO: replace explicit stochastic gradient
	y = calculateStochasticGradient(w, g, X, z) - calculateStochasticGradient(wPrevious, g, X, z)

	return (s, y)

def solveSQN(f, g, X, z = None, w1 = None, M=10, L=1.0, beta=1):
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

	## TODO: Give dimensions!
	if w1 == None:
		w1 = np.zeros(2)
		
	# step sizes alpha_k
	alpha = lambda k: beta/(k+1)

	t = -1
	(nSamples, nFeatures) = np.shape(X)
	w=w1
	#Set wbar = wPrevious = 0
	wbar = np.zeros(w1.shape)
	wPrevious = wbar

	s, y = deque(), deque()

	for k in itertools.count():
		if hasTerminated(f ,y ,w ,k):
			break

		X_S, y_S = chooseSample(nSamples, X, z)
		print "X_S, y_S:", X_S, y_S
		
		grad = calculateStochasticGradient(w, g, X_S, y_S)
		print "grad", grad
		wbarPrevious = wbar
		wPrevious = w
		wbar = wbar + w
		
		if k <= 2*L:
			w = w - alpha(k)*grad
		else:
			w = w - alpha(k)*getH(s,y).dot(grad)
		print "w: ", w
		#compute Correction pairs every L iterations
		if k%L == 0:
			t=t+1
			wbar = wbar/float(L)
			if t>0:
			#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH = chooseSample(nSamples, X, z)
				(s_t, y_t) = correctionPairs(w, wPrevious, g, X_SH, y_SH)
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft() 
			wbar = 0

	return w

