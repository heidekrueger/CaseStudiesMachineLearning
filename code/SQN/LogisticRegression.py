import numpy as np
import math
from sklearn import datasets

class LogisticRegression():
	"""
		Returns objective function F, sample loss function f and gradients g
		with respect to parameters w

		Parameters:
			X: Sample matrix (nSamples * nFeatures)
			y: label vector
	"""

	def __init__(self):
		self.sigmoid = lambda z: 1/(1.0+np.exp(-z))
		self.h = lambda w, X: self.sigmoid(np.dot(X,w))

	#define sigmoid function
	

	#define hypothesis function for ith sample
	

	def f(self, w, X, y):
		"""
		Loss functions as column vector
		"""
		print "testing:"
		print y
		print "  "
		hyp = self.h(w, X)
		return -y.dot(np.log(hyp))-(1-y).dot(np.log(1-hyp))


	def F(self, w, X, y):
		return self.f(w, X, y).sum()/float(X.shape[0])

	def g(self, w, X, y):
		hyp = self.h(w, X)
		return np.dot(X.T, hyp-y)/X.shape[0]

class LogisticRegressionTest(LogisticRegression):
	def __init__(self):
		LogisticRegression.__init__(self)

	def testF(self):
		iris = datasets.load_iris()
		X, y = iris.data[:5,:], iris.target[:5]

		w= np.zeros([X.shape[1],1])

		print "f complete"
		print self.f(w, X, y)
		print "f for first entry"
		print self.f(w, X[0,:], y[0])
		print "F"
		print self.f(w,X,y)
		print "g complete"
		print self.g(w, X, y)
