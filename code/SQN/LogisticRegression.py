import numpy as np
import math
from sklearn import datasets
import sklearn as sk

class LogisticRegression():
	"""
		Class representing LogisticRegression
	"""

	def __init__(self):
		# sigmoid function
		self.sigmoid = lambda z: 1/(1.0+np.exp(-z))
		# hypothesis function
		self.h = lambda w, X: self.sigmoid(np.dot(X,w))

	def f(self, w, X, y):
		"""
		Loss functions as column vector
		"""
		if np.isscalar(y):
			y = np.array([y])
		hyp = self.h(w, X)
		return -y*np.log(hyp)- (1-y)*(np.log(1-hyp))

	def F(self, w, X, y):
		"""
		Overall objective function
		"""
		return self.f(w, X, y).sum()/float(X.shape[0])

	def g(self, w, X, y):
		"""
		Gradient of F
		"""
		hyp = self.h(w, X)
		return np.dot(X.T, hyp-y)/X.shape[0]

class LogisticRegressionTest(LogisticRegression):
	def __init__(self):
		LogisticRegression.__init__(self)

	def testF(self):
		iris = datasets.load_iris()
		X, y = iris.data[:5,:], iris.target[:5,np.newaxis]
		w= np.zeros([X.shape[1],1])

		print "f complete"
		print self.f(w, X, y)
		print "f for first entry"
		print self.f(w, X[0,:], y[0])
		print "F"
		print self.F(w,X,y)
		print "g "
		print self.g(w, X, y)
