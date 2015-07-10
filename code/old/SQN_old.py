
import stochastic_tools
from stochastic_tools import stochastic_gradient
from stochastic_tools import armijo_rule

from SQN import getH, correctionPairs


def hasTerminated(f, grad, w, k, max_iter = 1e4, debug=False):
	"""
	Checks whether the algorithm has terminated

	Parameters:
		f: function for one sample
		y: difference of gradients
		w: current variable
		k: current iteration

	"""
	if debug:
		print "Check termination"
		print "len grad:", np.linalg.norm(grad) 
		#print "fun val", f(w)
	eps = 1e-6
	if k > max_iter:
		return True
	elif len(grad) > 0 and np.linalg.norm(grad) < eps:
		return True
	else:
		return False



def solveSQN_old(f, g, X, z = None, w1 = None, dim = None, iterator = None, M=10, L=1.0, beta=1, batch_size = 1, batch_size_H = 1, max_iter = 1e4, debug = False, sampleFunction = None):
	"""
	Parameters:
		f:= f_i = f_i(omega, x, z[.]), loss function for one sample. The goal is to minimize
			F(omega,X,z) = 1/nSamples*sum_i(f(omega,X[i,:],z[i]))
			with respect to w
		g:= g_i = g_i(omega, x, z), gradient of f

		X: nSamples * nFeatures numpy array of Data
		z: nSamples * 1 numpy array of targets

		w1: initial w

		M: Memory-Parameter
	"""
	assert M > 0, "Memory Parameter M must be a positive integer!"
	assert w1 != None or dim != None or iterator != None, "Please privide either a starting point or the dimension of the optimization problem!"
	

	# dimensions
	nSamples = len(X)
	nFeatures = len(X[0])
	
	input_iterator = False
	if w1 is None and dim is None:  
	    input_iterator = True
	    w1 = stochastic_tools.iter_to_array(iterator)
	elif w1 is None:
	    w1 = np.zeros(dim)
	#    w1[0] = 3
	#    w1[0] = 4
	w = w1

	if sampleFunction != None:
		chooseSample = sampleFunction
	else:
		chooseSample = stochastic_tools.sample_batch

	#Set wbar = w_previous = 0
	wbar = w1
	w_previous = w
	if debug: print w.shape
	# step sizes alpha_k
	alpha_k = beta
	#alpha = lambda k: beta/(k + 1)

	s, y = deque(), deque()
	
	# accessed data points
	t = -1
	H = None
	for k in itertools.count():
		
		# Draw mini batch
		X_S, z_S= chooseSample(w=w, X=X, z=z, b = batch_size)
		if debug: print "sample:", chooseSample
		if len(X_S) == 0:
			X_S, z_S= chooseSample(w=w, X=X, z=z, b = batch_size)
		# Check Termination Condition
		if debug: print "Iteration", k
		if len(X_S) == 0 or hasTerminated(f , stochastic_gradient(g, w, X_S, z_S) ,w ,k, max_iter = max_iter, debug=True):
			iterations = k
			break
		
		# Determine search direction
		if k <= 2*L:  	search_direction = -stochastic_gradient(g, w, X_S, z_S)
		else:	   	search_direction = -H.dot(stochastic_gradient(g, w, X_S, z_S))
		if debug: 		print "Direction:", search_direction.T
	
		# Compute step size alpha
		f_S = lambda x: f(x, X_S, z_S) if z is not None else f(x, X_S)
		g_S = lambda x: stochastic_gradient(g, x, X_S, z_S)
		alpha_k = armijo_rule(f_S, g_S, w, search_direction, start = beta, beta=.5, gamma= 1e-2 )
		alpha_k = max([alpha_k, 1e-5])
		    
		if debug: print "f\n", f_S(w)
		if debug: print "w\n", w
		if debug: print "alpha", alpha_k
		
		# Perform update
		w_previous = w
		w = w + np.multiply(alpha_k, search_direction)
		wbar += w
		# compute Correction pairs every L iterations
		if k%L == 0:
			t += 1
			wbar /= float(L) 
			if t>0:
				#choose a Sample S_H \subset [nSamples] to define Hbar
				X_SH, y_SH = chooseSample(w, X, z, b = batch_size_H)
				
				(s_t, y_t) = correctionPairs(g, w, w_previous, X_SH, y_SH)
				
				if debug: print "correction shapes", s_t, y_t
				s.append(s_t)
				y.append(y_t)
				if len(s) > M:
					s.popleft()
					y.popleft()
				
				H = getH(s, y)
				
			wbar = np.zeros(dim)

	if iterations < max_iter:
		print "Terminated successfully!" 
	print "Iterations:\t\t", iterations
	
	if input_iterator:  
		stochastic_tools.set_iter_values(iterator, w)
		return iterator
	else:
		return w
		
