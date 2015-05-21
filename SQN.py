import numpy as np
import scipy as sp


def solveSQN(f,grad,hess,X,y=None,w1,M=1,L=1,beta=1):
	"""
	Parameters:
		f: f(omega, x, y[.]), loss function for one sample. The goal is to minimize
			F(omega,X,y) = 1/nSamples*sum_i(f(omega,X[i,:],y[i]))
			with respect to w
		grad: g(omega, x, y), gradient of f
		hess: hess(omega,x,y), hessian of f

		X: nSamples * nFeatures numpy array of Data
		y: nSamples * 1 numpy array of targets

		w1: initial w

		M: Memory-Parameter
	"""
	t = -1
	nSamples, nFeatures = X.shape

	# step sizes alpha_k
	alpha = lambda k: beta/k
