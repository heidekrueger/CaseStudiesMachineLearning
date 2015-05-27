import numpy as np
import math

def LogisticRegression(X,y):
	"""
		Returns objective function F, sample loss function f and gradients g
		with respect to parameters w

		Parameters:
			X: Sample matrix (nSamples * nFeatures)
			y: label vector
	"""

	#define sigmoid function
	sigmoid = lambda z: 1/(1.0+np.exp(-z))

	#define hypothesis function for ith sample
	h = lambda w: sigmoid(np.dot(X,w))

	def f(w):
		"""
		Loss functions as column vector
		"""
		hyp = h(w)
		return -y*np.log(hyp)-(1-y)*np.log(1-hyp)


	F  = lambda w: f(w).sum()/X.shape[0]