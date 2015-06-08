
import SQN, SQN_LAZY
import numpy as np

'''
The Rosenbrock function:

f(x, y) = (a-x)^2 + b(y-x^2)^2

It has a global minimum at 
(x, y)=(a, a^2), 
where f(x, y)=0. 

Usually a = 1 and b = 100.

Source: https://en.wikipedia.org/wiki/Rosenbrock_function
'''

def test_rosenbrock(sqn_method):

	X = np.asarray([ np.zeros(4) for i in range(4) ])
	
	a = 1
	b = 100
	rosenbrock = lambda x, X: (a - x[0])**2 + b*(x[1] - x[0]**2)**2

	rosengrad = lambda x, X: np.asarray([
			2*(a-x[0])*(-1) + 2*(x[1]-x[0]**2)*(-2*x[1]), 
			2*(x[1]-x[0]**2)
									])
	print sqn_method(rosenbrock, rosengrad, X=X, z=None, w1 = None, dim = 2, M=10, L=1.0, beta=0.1)


from LogisticRegression import LogisticRegression
def test_Logistic_Regression(sqn_method):
	logreg = LogisticRegression()
	
	func = lambda w, X, z: logreg.F(w, X, z)
	grad = lambda w, X, z: logreg.g(w, X, z)
	
	print sqn_method(func, grad, X=X, z=None, w1 = None, dim = 2, M=10, L=1.0, beta=0.1)
    
if __name__ == "__main__":
	print "\nSQN:"
	test_rosenbrock(SQN.solveSQN)
	print "\nLazy SQN:"
	test_rosenbrock(SQN_LAZY.solveSQN)
	
	print "Logistic Regression: SQN"
	test_Logistic_Regression(SQN.solveSQN)
	
	print "Logistic Regression: Lazy SQN"
	test_Logistic_Regression(SQN_LAZY.solveSQN)
	
	