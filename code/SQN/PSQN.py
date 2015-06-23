import stochastic_tools
from stochastic_tools import stochastic_gradient
from stochastic_tools import armijo_rule
from SQN import SQN

class PSQN(SQN):
	"""
	Proximal Methods SQN
	"""

	def __init__(self):
		SQN.__init__(self)
		self.options['L'] = 1
		self.options['M'] = 1
		self.w_previous = None
		# TODO: Add further options
		    
	def _get_search_direction(self, g_S):
		"""
			TODO: Add proximal stuff in order to get search direction
			B, H, u, d = compute_sr1_update(self.s, self.y, k, **options)
			temp_x_new = compute_proximal(B, H, u, d, g_S, self.w)
			search_direction = temp_x_new - self.w
		"""
		raise NotImplementedError
		
		search_direction = -g_S(self.w)
		if self.debug: 
			print "Direction:", search_direction.T
		
		return search_direction
	    
	def _update_correction_pairs(self, g, X, z):
		
		# draw hessian sample and get the corresponding stochastic gradient
		X_SH, y_SH = self._draw_sample(X, z, self.options['batch_size_H'])
		g_SH = lambda x: stochastic_gradient(g, x, X_SH, y_SH)
			    
		s_t = self.wbar - self.wbar_previous
		y_t = g_SH(self.wbar) - g_SH(self.wbar_previous)
			
		if abs(y_t).sum() != 0:
		    self.s.append(s_t)
		    self.y.append(y_t)
		else:
		    print "PROBLEM! zero y"
		    
		if len(self.s) > self.options['M']:
			self.s.popleft()
			self.y.popleft()
			
		# TODO:: Check if the updating of s and y is correctly done
		raise NotImplementedError
		return
	