

import SQN
import numpy as np

def test_rosenbrock():

	X = np.asarray([ np.zeros(4) for i in range(4) ])
	
	a = 1
	b = 100
	rosenbrock = lambda x, X: (a - x[0])**2 + b*(x[1] - x[0]**2)**2

	rosengrad = lambda x, X: np.asarray([
			2*(a-x[0])*(-1) + 2*(x[1]-x[0]**2)*(-2*x[1]), 
			2*(x[1]-x[0]**2)
									])

	print SQN.solveSQN(rosenbrock, rosengrad, X=X, z=None, w1 = None, M=10, L=1.0, beta=1)

if __name__ == "__main__":
	test_rosenbrock()