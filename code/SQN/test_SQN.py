
import SQN, SQN_LAZY
import numpy as np
import timeit

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
			2*(x[1]-x[0]**2)
									])
	print sqn_method(rosenbrock, rosengrad, X=X, z=None, w1 = None, dim = 2, M=10, L=1.0, beta=0.1)


from LogisticRegression import LogisticRegression_1D
import datasets
def test_Logistic_Regression(sqn_method, X, z, w1 = None, dim = 3, M=10, L=5, beta=0.1, batch_size = 5, batch_size_H = 10, max_iter=300, sampleFunction = "logreg", debug = False):
	logreg = LogisticRegression_1D()
	func = lambda w, X, z: logreg.F(w, X, z)
	grad = lambda w, X, z: logreg.g(w, X, z)

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
		w = sqn_method(func, grad, X=X, z=z, w1 = w1, dim = dim, M=M, L=L, beta=beta, batch_size = batch_size, batch_size_H = batch_size_H, max_iter=max_iter, sampleFunction = sampleFunction, debug = False)
		results.append(func(w, X, z))
		print results[-1]

	print "time: ", (timeit.default_timer()-t)
	print "avg objective:", sum(results)/N

    
if __name__ == "__main__":
	
	X, z = datasets.load_data1()
	
	print "\nSQN:"
	#test_rosenbrock(SQN.solveSQN, X, z)
	#print "\nLazy SQN:"
	#test_rosenbrock(SQN_LAZY.solveSQN, X, z)
	
	
	print "Logistic Regression: SQN"
	test_Logistic_Regression(SQN.solveSQN, X, z )
	
	#print "Logistic Regression: Lazy SQN"
	#test_Logistic_Regression(SQN_LAZY.solveSQN, X, z)
