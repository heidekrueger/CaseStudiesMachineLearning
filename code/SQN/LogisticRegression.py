import numpy as np
import math
from sklearn import datasets
import sklearn as sk

class LogisticRegression():
	"""
		Class representing LogisticRegression
		
	"""

	def __init__(self, lam = 0.):
		'''
		
		lam = L2 regularization parameter
		    
		'''
		# hypothesis function
		self.expapprox = 30
		self.lam = lam
	# sigmoid function
		
	def sigmoid(self, z):
	    #print "sigmoid\n", z
	    for i in range(len(z)):
		if z[i] > self.expapprox:
		    z[i] = self.expapprox
		elif z[i] < -self.expapprox:
		    z[i] = -self.expapprox
	    return 1/(1.0+np.exp(-z))
	
	def h(self, w, X): 
	    if len(X.shape)==1:
		#X.shape = (1, X.shape[0])
	        X = np.atleast_2d(X)
	        #print 'X' , X
	    return self.sigmoid(np.dot(X,w))
		
	def f(self, w, X, y):
		"""
		Loss functions as column vector
		"""
		#print "X",  X
		#print "w", w
		if np.isscalar(y):
			y = np.array([y])
		hyp = self.h(w, X)
		#print "f shape", np.shape(hyp), np.shape(X), np.shape(w)
		#print "hyp", hyp
		return -y*np.log(hyp)- (1-y)*(np.log(1-hyp))

	def F(self, w, X, y, lam = 1):
		"""
		Overall objective function
		"""
		#return sum([self.f(w,X[i,:],y[i]) for i in range(X.shape[0])]) /float(X.shape[0]) + 0.5 * self.lam * ( np.inner(w[1:], w[1:]) )/X.shape[0] 
		return self.f(w, X, y).sum()/float(X.shape[0]) + 0.5 * self.lam * ( np.inner(w[1:], w[1:]) )/X.shape[0]

	def g(self, w, X, y, lam = 1):
		"""
		Gradient of F
		"""
		hyp = self.h(w, X)
		#print "HYP", hyp.shape
		if len(hyp) == 1:
		    hyp = hyp[0,0]
		    y = y[0]
		    return np.multiply((hyp - y)/X.shape[0], X) + np.multiply(self.lam/X.shape[0], w).T
		else:
		    print (np.dot( X.T, (hyp-y[:,np.newaxis]) )/X.shape[0]).shape
		    print (np.multiply( self.lam/float(X.shape[0]), w)).shape
		    return np.dot( X.T, (hyp-y[:,np.newaxis]) )/X.shape[0] + np.multiply( self.lam/float(X.shape[0]), w)















class LogisticRegression_1D():
	"""
		Class representing LogisticRegression
		
	"""

	def __init__(self, lam = 0.):
		'''
		
		lam = L2 regularization parameter
		    
		'''
		# hypothesis function
		self.expapprox = 30
		self.lam = lam
	# sigmoid function
		
	def sigmoid(self, z):
	    #print "sigmoid\n", z
	    if z > self.expapprox:
		z = self.expapprox
	    elif z < -self.expapprox:
		z = -self.expapprox
	    return 1/(1.0+np.exp(-z))
	
	def h(self, w, X): 
	    #print w.shape, X.shape, np.dot(X,w).shape
	    return self.sigmoid(np.multiply(w, X).sum())
	    
	def f(self, w, X, y):
		"""
		Loss functions as column vector
		"""
		#print "X",  X
		#print "w", w
		hyp = self.h(w, X)
		return -y* np.log(hyp)- (1-y)*(np.log(1-hyp))
	    
	def F(self, w, X, y, lam = 0.0):
		"""
		Overall objective function
		"""
		return sum([self.f(w,X[i],y[i]) for i in range(len(y))]) /float(len(y)) + 0.5 * self.lam * ( np.linalg.norm(w[1:])**2 ) /float(len(y))
		
	def g(self, w, X, y, lam = 0.0):
		"""
		Gradient of F
		"""
		hyp = self.h(w, X)
		return ((hyp - y)/float(len(y)))* X + (self.lam/float(len(y)))* w
		
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

