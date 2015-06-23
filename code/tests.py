

"""
Logistic Regression
"""
from SQN.LogisticRegression import LogisticRegression
from SQN.LogisticRegressionTest import LogisticRegressionTest


"""
SQN
"""
import SQN.SQN as SQN
import numpy as np
import timeit

from SQN import stochastic_tools
'''
The Rosenbrock function:

f(x, y) = (a-x)^2 + b(y-x^2)^2

It has a global minimum at 
(x, y)=(a, a^2), 
where f(x, y)=0. 

Usually a = 1 and b = 100.

Source: https://en.wikipedia.org/wiki/Rosenbrock_function
'''

def test_rosenbrock(sqn_method, X, z):

	X = np.asarray([ np.zeros(4) for i in range(4) ])
	
	a = 1
	b = 100
	rosenbrock = lambda x, X: (a - x[0])**2 + b*(x[1] - x[0]**2)**2

	rosengrad = lambda x, X: np.asarray([
			2*(a-x[0])*(-1) + 2*(x[1]-x[0]**2)*(-2*x[1]), 
			2*(x[1]-x[0]**2) ])
									
	print sqn_method(rosenbrock, rosengrad, X=X, z=None, w1 = None, dim = 2, M=10, L=1.0, beta=0.1)


def test_Logistic_Regression(sqn_method, X, z, w1 = None, dim = 3, M=10, L=5, beta=0.1, batch_size = 5, batch_size_H = 10, max_iter=300, sampleFunction = "logreg", debug = False):
	logreg = LogisticRegression()
	func = lambda w, X, z: logreg.F(w, X, z)
	grad = lambda w, X, z: logreg.g(w, X, z)
	L = 1e4
	print "M:", M
	print "L:", L
	print "batch_size", batch_size
	print "batch_size_H", batch_size_H
	print "max_iter", max_iter
	print "sampleFunction", sampleFunction

	print

	if sampleFunction=="logreg":
		sampleFunction = logreg.sample_batch
	else:
		sampleFunction = None

	results = []
	N = 10

	t = timeit.default_timer()
	for i in range(N):
		print i, "th iteration"
		w = sqn_method(func, grad, X=X, z=z, w1 = w1, dim = dim, M=M, L=L, beta=beta, batch_size = batch_size, batch_size_H = batch_size_H, max_iter=max_iter, sampleFunction = sampleFunction, debug = True)
		results.append(func(w, X, z))
		print results[-1]

	print "time: ", (timeit.default_timer()-t)
	print "avg objective:", sum(results)/N



""" 
Dictionary Learning
"""







"""
Proximal Methods
"""









"""
Main
"""
import data.datasets as datasets


import sys
if __name__ == "__main__":
	
	testcase = 1
	
	if len(sys.argv) > 1:
		testcase = sys.argv[1]
	
	print "Using testcase", testcase
	if testcase == '1':
		X, z = datasets.load_data1()
		print "\nSQN:"
		test_rosenbrock(SQN.solveSQN, X, z)
		#print "\nLazy SQN:"
		#test_rosenbrock(SQN_LAZY.solveSQN, X, z)
	elif testcase == '2':
		X, z = datasets.load_data1()
		print "Logistic Regression: SQN"
		test_Logistic_Regression(SQN.solveSQN, X, z )
		#print "Logistic Regression: Lazy SQN"
		#test_Logistic_Regression(SQN_LAZY.solveSQN, X, z)
	elif testcase == '3':
		X, z = datasets.load_iris()
		print "Logistic Regression using SQN"
		logregtest = LogisticRegressionTest()
		logregtest.test_classification(X, z)
	elif testcase == '4':
		X, z = datasets.load_data1()
		z = None
		a = 1
		b = 100
		rosenbrock = lambda x, X: (a - x[0])**2 + b*(x[1] - x[0]**2)**2

		rosengrad = lambda x, X: np.asarray([
				2*(a-x[0])*(-1) + 2*(x[1]-x[0]**2)*(-2*x[1]), 
				2*(x[1]-x[0]**2) ])
		print "\nSQN:"
		sqn = SQN.SQN()
		sqn.set_options({'dim':2, 'L': 5})
		sqn.solve(rosenbrock, rosengrad, X, z)
		
	elif testcase == '5':
		X, z = datasets.load_data1()
		logreg = LogisticRegression()
		#func = lambda w, X, z: logreg.F(w, X, z)
		#grad = lambda w, X, z: logreg.g(w, X, z)

		print "\nSQN class:"
		sqn = SQN.SQN()
		#sqn.debug = True
		sqn.set_options({'dim':len(X[0]), 'max_iter': 1600, 'batch_size': 20, 'beta': 10, 'batch_size_H': 10, 'L': 3, 'sampleFunction':logreg.sample_batch})
		sqn.solve(logreg.F, logreg.g, X, z)

	elif testcase == 'higgs':
		"""Runs SQN-LogReg on the Higgs-Dataset, 
		which is a 7.4GB csv file for binary classification
		that can be obtained here:
		https://archive.ics.uci.edu/ml/datasets/HIGGS
		the file should be in <Git Project root directory>/datasets/
		"""
		rowlim = 1000
		X, z = datasets.load_higgs(rowlim)

		logreg = LogisticRegression()
		func = lambda w, X, z: logreg.F(w, X, z)
		grad = lambda w, X, z: logreg.g(w, X, z)

		print "\nSQN, Higgs-Dataset, #rows:", rowlim
		sqn = SQN.SQN()
		sqn.set_options({'dim':29, 'N':5e5, 'max_iter': 1e3, 'batch_size': 10, 'batch_size_H': 10, 'L':3, 'beta':10, 'M':100})
		sqn.solve(func, grad, X, z)
		
	elif testcase == 'higgs2':
		"""Runs SQN-LogReg on the Higgs-Dataset, 
		which is a 7.4GB csv file for binary classification
		that can be obtained here:
		https://archive.ics.uci.edu/ml/datasets/HIGGS
		the file should be in <Git Project root directory>/datasets/
		"""
		logreg = LogisticRegression()
		func = lambda w, X, z: logreg.F(w, X, z)
		grad = lambda w, X, z: logreg.g(w, X, z)

		print "\nSQN, Higgs-Dataset"
		sqn = SQN.SQN()
		#sqn.debug = True
		sqn.set_options({'dim':29, 'sampleFunction': stochastic_tools.sample_batch_higgs, 'N':5e5, 'max_iter': 1e3, 'batch_size': 10, 'batch_size_H': 10, 'L':3, 'beta':10, 'M':100})
		
		sqn.set_start(dim=29)
		X, z = stochastic_tools.sample_batch_higgs(None, N=5e5, b=5e5)
		import itertools
		for k in itertools.count():
		    
			w = sqn.solve_one_step(func, grad, k=k)
			if k % 5 == 0:
				sqn.options['batch_size'] += 10
				sqn.options['batch_size_H'] += 10
				print logreg.F(w, X, z)
			if k > sqn.options['max_iter'] or sqn.termination_counter > 4:
			    iterations = k
			    break
			
		#w = sqn.solve(func, grad)
		
	elif testcase == 'prox':
		#a = 1
		#b = 100
		#rosenbrock = lambda x: (a - (x[0]+1))**2 + b*(x[1]+1 - (x[0]+1)**2)**2
		#rosengrad = lambda x: np.asarray([2*(a-x[0]-1)*(-1) + 2*(x[1]-(x[0]+1)**2)
		#                                    *(-2*(x[1]+1)), 2*(x[1]-(x[0]+1)**2)])
		
		def f(x):
		    return x**2
		def grad_f(x):
		    return 2*x
		    
		x0 = np.array([13,4])
		
		# x, k = compute_0sr1(f, grad_f, x0, algo=2, lower_b=np.array([0,0]), upper_b=np.array([100,400]))
		
		x, k = compute_0sr1(f, grad_f, x0)
		
		print(x)
		print(k)
		
	elif testcase == '66':
		from data.datasets import split_into_files
		split_into_files('../datasets/HIGGS.csv', '../datasets/HIGGS/')