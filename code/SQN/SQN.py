import numpy as np
import scipy as sp
import random as rd
import itertools
from collections import deque

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

def calculateStochasticGradient(w, g, X, y=None):
	"""
	Calculates Stochastic gradient of F at w as per formula (1.4)
	"""
	nSamples, nFeatures = X.shape
	if y is None:
		return sum([g(w,X[i,:]) for i in range(X.shape(0))])
	else:
		assert(len(X)==len(y), "Error: Dimensions mus match")
		return sum([g(w,X[i,:],y[i]) for i in range(X.shape(0))])


def getH(s, y):
	"""
	returns H_t as defined in algorithm 2
	"""
	assert(len(s)>0, "s cannot be empty.")
	assert(len(s)==len(y), "s and y must have same length")
	assert(s[0].shape == y[0].shape, "s and y must have same shape")
	assert(y[-1].abs().sum() != 0, "last y entry cannot be 0!")
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

def correctionPairs(w, wPrevious, g, X, y):
	"""
	returns correction pairs s,y

	"""
	s = w-wPrevious
	#TODO: replace explicit stochastic gradient
	y = calculateStochasticGradient(w, g, X, y) - calculateStochasticGradient(wPrevious, g, X, y)

	return (s, y)

def solveSQN(f, g, X, y=None, w1, M=10, L=1.0, beta=1):
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
	assert(M>0, "Memory Parameter M must be a positive integer!")


	# step sizes alpha_k
	alpha = lambda k: beta/k

	t = -1
	nSamples, nFeatures = X.shape
	w=w1
	#Set wbar = wPrevious = 0
	wbar = np.zeros(w1.shape)
	wPrevious = wbar

	s, y = deque(), deque()


	for k in itertools.count():
		if hasTerminated(f,g,w,k):
			break

		S = chooseSample(nSamples)
		grad = calculateStochasticGradient(w, g, X[S,:], y[S])
		wbarPrevious = wbar
		wPrevious = w
		wbar = wbar + w
		if k <= 2*L:
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
				(s_t, y_t) = correctionPairs(w, wPrevious, X[sampleH,:], y[sampleH])
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft() 
			wbar = 0

	return w


if __name__ == "__main__":
	rosenbrock = lambda x: (1-x[0])**2+100*(x[1]-x[0]**2)**2
	rosengrad = 