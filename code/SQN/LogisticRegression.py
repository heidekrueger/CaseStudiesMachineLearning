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
		Loss functions
		"""
		hyp = h(w)
		return -np.dot(np.log(hyp),y)-np.dot(math.log(1-hyp),(1-y))