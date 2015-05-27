import numpy as np
import scipy as sp
import random as rd
import itertools

def hasTerminated(f,g,w,k):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		g: gradient entry for one sample
		w: current variable
		k: current iteration

	"""
	eps = 1e-4
	if g(w)<eps or k>100:
		return True
	else:
		return False

def chooseSample(N,b=.2):
	"""
	returns a subset of [N] as a list?

	Parameters:
		N: Size of the original set
		b: parameter for subsample size (e.g. b=.1)
	"""
	return rd.sample(range(N), int(b*N))

def calculateStochasticGradient(w):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	raise NotImplementedError

def getH(t):
	"""
	returns H_t as defined in algorithm 2
	"""
	raise NotImplementedError 

def correctionPairs(w,wPrevious):
	"""
	returns correction pairs s,y
	"""
	raise NotImplementedError

def solveSQN(f,grad,X,y=None,w1,M=1,L=1.0,beta=1):
	"""
	Parameters:
		f:= f_i = f_i(omega, x, y[.]), loss function for one sample. The goal is to minimize
			F(omega,X,y) = 1/nSamples*sum_i(f(omega,X[i,:],y[i]))
			with respect to w
		g:= g_i = g_i(omega, x, y), gradient of f

		X: nSamples * nFeatures numpy array of Data
		y: nSamples * 1 numpy array of targets

		w1: initial w

		M: Memory-Parameter
	"""
	# step sizes alpha_k
	alpha = lambda k: beta/k

	t = -1
	nSamples, nFeatures = X.shape
	w=w1
	#Set wbar = wPrevious = 0
	wbar = np.zeros(w1.shape)
	wPrevious = wbar

	for k in itertools.count():
		if hasTerminated(f,g,w,k):
			break

		S = chooseSample(nSamples)
		grad = calculateStochasticGradient(w)
		wbarPrevious = wbar
		wPrevious = w
		wbar = wbar +w
		if k<= 2*L:
			w = w - alpha(k)*grad
		else:
			w = w-alpha(k)*getH(t)*grad

		#compute Correction pairs every L iterations
		if k%L == 0:
			t=t+1
			wbar = wbar/float(L)
			if t>0:
			#choose a Sample S_H \subset [nSamples] to define Hbar
				sampleH = chooseSample(nSamples)
				s,y = correctionPairs(w,wPrevious)
			wbar = 0

	return w
