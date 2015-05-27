
import numpy as np
import scipy.optimize, scipy.linalg

doc = '''
This is an Quasi-Newton implementation adapted from 
Optimization, Theory and Practice
by Mohan C Joshi and Kannan M Moudgalya, 
Alpha Science International LTD, 2008, Mumbay, India
'''


##
## Finite difference quotient for vectos
##
def finite_differences(f, x, h = 1e-5):

    g = np.array([0.0]*len(x))
    x_tmp = np.array([0.0]*len(x))
    
    for i in range(len(x)):
        np.copyto(x_tmp, x)
        x_tmp[i] += h
        g[i] = ( f(x) - f(x_tmp) ) / h

    return g

##
## limit some quotient
##
def ensure_quotient(q, eps):
    if q >= 0:
        q = max(q, eps)
    else:
        q = min(q, -eps)
    return q

##
## DFP Update of the Inverse
##  M_{k+1} = M_k + \alpha_k \frac{d_k d_k^T}{\delta g_k^T d_k} - \frac{p_k \delta g_k^T}{p_k^T\delta g_k^T} M_k
##
def Update_Inverse_DFP(M_k, alpha_k, d_k, dg_k):
    ##
    ## get difference of search directions for conjugating
    ##
    p_k = M_k.dot(dg_k)
    
    ##
    ## get quotients and limit them so that there is no devision by zero
    ##
    q_1 = np.inner(d_k, dg_k)
    q_1 = ensure_quotient(q_1, 1e-8)
    q_2 = np.inner(p_k, dg_k)
    q_2 = ensure_quotient(q_2, 1e-8)

    ##
    ## calculate update
    ##
    M_new = M_k + alpha_k * np.outer(d_k, d_k) / q_1 - np.outer(p_k, dg_k).dot(M_k) / q_2
    
    return M_new

##
## BFGS Update of the Inverse
##
def Update_Inverse_BFGS(M_k, dx_k, dg_k):
    
    ##
    ## get quotients and limit them so that there is no devision by zero
    ##
    q = np.inner(dx_k, dg_k)
    q = ensure_quotient(q, 1e-98) 

    ##
    ## get difference of search directions for conjugating
    ##
    U = np.outer(dx_k, dg_k)
    
    ##
    ## Identity
    ##
    assert U.shape[0] == U.shape[1], "Matrix dimension mixed up!"
    I = np.identity(np.shape(U)[0])
    
    ##
    ## calculate update
    ##
    M_new = ( I - U/q ).dot(M_k).dot( I - U.T/q ) + np.outer(dx_k, dx_k) / q
    
    return M_new

##
## perform a step of a quasi newton's method
##     
def Quasi_Newton_Step(f, x_k, g_k, M_k, h = 1e-6, grad=lambda x: finite_differences(f, x), Update_Inverse = Update_Inverse_BFGS):
    
    ##
    ## determine direction
    ## d_k = - M_k g_k
    ##
    d_k = -M_k.dot(g_k)
    
    ##
    ## do golden rationline search to find an optimal alpha
    ## \alpha_k = argmin_\alpha \{  f(x_k + \alpha d_k)  \}
    ##
    line = lambda alpha: f( x_k + alpha * d_k )
    
    alpha_k = scipy.optimize.golden( line )

    ##
    ## get update x
    ## x_{k+1} = x_k + \alpha_k d_k
    ##
    x_new = x_k + alpha_k * d_k
    
    ##
    ## get gradient at new position
    ## g_{k+1} = \nabla f(x_{k+1})
    ##
    g_new = grad(x_new)
    
    ##
    ## get difference of gradients
    ##
    dg_k = g_new - g_k
    
    ##
    ## get difference of values
    ##
    dx_k = x_new - x_k
    
    ##
    ## get new M matrix
    ##
    M_new = Update_Inverse(M_k, dx_k, dg_k)
    
    return([x_new, g_new, M_new])


##
## Break condition for newton method
##
def Break_Condition(f, x_new, x_old, eps):
    return sum(abs(x_new-x_old))**2 < eps and abs(f(x_new) - f(x_old)) < eps 

##
## Quasi Newton algorithm
## 
## TODO: This implementation is very memory consuming because it keeps track of all iterates
## TODO: The precision is inherently dependent on the differencing step with h. What is the best value for that? eps?
def Quasi_Newton(f, x_0, max_iter = 100, h = 1e-6, eps = 1e-8, method = "BFGS"):


    if method == "BFGS":
        Update_Inverse = Update_Inverse_BFGS
    elif method == "DFP":
        Update_Inverse = Update_Inverse_DFP
    else:
        print "ERROR! Unknown Method: %s" %method
        return ["-", "-"], 0
    ##
    ## store all intermediate values in lists
    ##
    x, g, M = [], [], []
    
    ##
    ## starting configuration
    ##
    g_0 = finite_differences(f, x_0)
    M_0 = np.identity(len(g_0))
    
    ##
    ## append to lists
    ##
    x.append(x_0)
    g.append(g_0)
    M.append(M_0)
    
    ##
    ## start iterations
    ##
    for k in range(max_iter):        
        
        ##
        ## perform one quasi newton update step
        ##
    
        [ x_new, g_new, M_new ] = Quasi_Newton_Step(f, x[-1], g[-1], M[-1], h = h, Update_Inverse = Update_Inverse)
        
        ##
        ## append results of step to lists
        ##
        x.append(x_new)
        g.append(g_new)
        M.append(M_new)
        
        ##
        ## break condition
        ##
        if Break_Condition(f, x[-1], x[-2], eps):
            print "Converged after %d iterations" %(k+1)
            break
    if not Break_Condition(f, x[-1], x[-2], eps):
        print "Diverged after %d!" %max_iter
        
    return x[-1], f(x[-1])
  
##      
## Test
##
doc_himmelblau = ''' 
	Himmelblau's function.
	f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2.

	It has one local maximum at x = -0.270845 \, and y = -0.923039 \, where 
f(x,y) = 181.617 \,, and four identical local minima:

    f(3.0, 2.0) = 0.0, 
    f(-2.805118, 3.131312) = 0.0, 
    f(-3.779310, -3.283186) = 0.0, 
    f(3.584428, -1.848126) = 0.0. 

	source: https://en.wikipedia.org/wiki/Himmelblau%27s_function
'''

print doc_himmelblau

print doc

def himmelblau(x, y):
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

f = lambda x: himmelblau(x[0], x[1])

x_0 = np.array([6.0,0.0])

print "Starting Point:", x_0
print
x_opt, f_opt = Quasi_Newton(f, x_0, max_iter = 100, h = 1e-6, eps = 1e-6 )
print "Optimal value %0.2f found at" %f_opt, list(x_opt)
print
print scipy.optimize.fmin_bfgs(f, x_0)
        
        
