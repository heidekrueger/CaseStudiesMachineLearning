import numpy as np
import math
import sklearn as sk

import SQN


class LogisticRegression():
	"""
		Class representing LogisticRegression		
		Accepts LISTS of np.arrays ONLY!!!
	"""

	def __init__(self, lam_1 = 0., lam_2 = 0):
		'''
		:lam_1 = L1 regularization parameter    
		:lam_2 = L2 regularization parameter
		'''
		# hypothesis function
		self.expapprox = 30
		self.lam_1 = lam_1
		self.lam_2 = lam_2
		
		self.w = None
		# performance analysis
		self.fevals = 0
		self.gevals = 0
		self.adp = 0
		
	# sigmoid function
	def sigmoid(self, z):
	    if math.isnan(z):
		    z = 0
	    else:
		    z = np.sign(z) * min([np.abs(z), self.expapprox])
	    return 1/(1.0+np.exp(-z))
	
	def h(self, w, X): 
	#	print type(X)
	#	print type(X[0])
	#	print w
	#	print type(w)
	#	print "debug:", np.multiply(w,X)
		return self.sigmoid(np.multiply(w, X).sum())
	    
	def f(self, w, X, y):
		"""
		Loss functions as column vector
		"""
		hyp = self.h(w, X)
		self.fevals += 1
		return -y* np.log(hyp)- (1-y)*(np.log(1-hyp))
	    
	def L_2(self, w):
		return  0.5 * self.lam_2 * (np.linalg.norm(w[1:])**2)
	    
	def F(self, w, X, y, lam = 0.0):
		"""
		Overall objective function
		"""
		#return sum(map(lambda t: self.f(w, t[0], t[1]), zip(X, y)))/len(X) +  self.L_2(w) 
		return sum([self.f(w,X[i],y[i]) for i in range(len(y))]) /float(len(y)) + self.L_2(w) 

	def g(self, w, X, y, lam = 0.0):
		"""
		Gradient of F
		"""
		hyp = self.h(w, X)
		self.gevals += 1
		return (hyp - y)* X + self.lam_2* w
	
	def train(self, X, y, method='SQN'):
		'''
		Determine the regression variable w
		'''
		assert len(X) > 0, "ERROR: Need at least one sample!"
		assert len(X) == len(y), "ERROR: Sample and label list need to have same length!"
		
		
		if method == 'SQN':
			M = 10
			L = 10
			beta = 1.0
			batch_size = 10
			batch_size_H = 10
			max_iter = 1600
			sqn = SQN.SQN()
			#sqn.debug = True
			sqn.set_options({'dim':len(X[0]), 
						    'max_iter': 45, 
						    'batch_size': 10, 
						    'beta': 10., 
						    'M': 10,
						    'batch_size_H': 10, 
						    'L': 10,  'sampleFunction':self.sample_batch})
			self.w = sqn.solve(self.F, self.g, X=X, z=y)
			
			#self.w = SQN.solveSQN(self.F, self.g, X=X, z=y, w1 = None, dim = len(X[0]), M=M, L=L, beta=beta, batch_size = batch_size, batch_size_H = batch_size_H, max_iter=max_iter, sampleFunction = self.sample_batch)
			
		else:
			raise NotImplementedError("ERROR: Method %s not implemented!" %method)
	
	def predict(self, X):
		"""
		calculate the classification probability of samples
		:X A list of samples
		"""
		if len(np.shape(X)) < 2:
			X = [X]
		return map( lambda x: self.h(self.w, x), X)
		
	def sample_batch(self, w, X, z = None, b = None, r = None, debug = False):
		"""
		returns a subsample X_S, y_S of the data, choosing only datapoints
		that are currently misclassified

		Parameters:
			w: Regression variable
			X: training data
			z: Label
			b: parameter for desired max. subsample size (e.g. b=10)
			r: desired relative max. subsample size (e.g. r=.1)
		"""
		if debug: print "debug: ", b
		assert b != None or r!= None, "Choose either absolute or relative sample size!"
		assert (b != None) != (r!= None), "Choose only one: Absolute or relative sample size!"
		
		# determine factual batch size
		N = len(X)
		if b != None:
		    nSamples = b
		else:
		    nSamples = r*N
		if nSamples > N:
		    if debug: print "Batch size larger than N, using whole dataset"
		    nSamples = N

		# Find samples that are not classified correctly

		#TODO: 

		sampleList = []
		searchList = np.random.permutation(N)
		for i in searchList:
			if self.f(w, X[i],z[i]) > .1:
				sampleList.append(i)

			if len(sampleList) == nSamples: #found enough samples
				break

		# if not enough samples are found, we simply return a smaller sample!
		nSamples = len(sampleList)
		 
		X_S = np.asarray([X[i] for i in sampleList])
		z_S = None if z is None else np.array([z[i] for i in sampleList])
		# Count accessed data points
		self.adp += nSamples
		
		if debug: print X_S, z_S
		   
		if z == None or len(z) == 0:
			return X_S, None
		else: 
			return X_S, z_S

