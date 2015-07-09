import numpy as np
import itertools


def get_H(s,y, v):
        
        I = np.identity(len(s[0]))
        H = np.dot((np.inner(s[-1], y[-1]) / np.inner(y[-1],
                   y[-1])), I)
        
        for (s_j, y_j) in itertools.izip(s, y):
            print "H*v:", H.dot(v)
            rho = 1.0/np.inner(s_j, y_j)
            V = I - rho * np.outer(s_j, y_j)
            H = V.dot(H).dot(V.T)
            H += rho * np.outer(s_j, s_j)
            
        return H

def two_loop_recursion(s, y, v):
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = v
        a = []
        
        for j in reversed(range(len(s))):
            rho = 1.0 / np.inner(y[j], s[j])
            alpha = rho * np.inner(s[j], q)
            q = q - np.multiply(alpha, y[j])
            a.append( alpha )
            
        H_k = np.inner(y[-2], s[-2]) / np.inner(y[-2], y[-2])
        z = np.multiply(H_k, q)
        
        for i in range(len(s)):
            rho = 1.0 / np.inner(y[i], s[i])

            b_i = rho * np.inner(y[i], z)
            z = z + np.multiply(a[i] - b_i, s[i])
        return z


def two_loop_recursion_zip(s, y, v):
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = v
        a = []
        for s_j, y_j in reversed(zip(s,y)):
            alpha = 1.0 * np.inner(s_j, q) / np.inner(s_j, y_j)
            q = q - np.multiply(alpha, y_j)
            a.append( alpha )
            
        H_k = np.inner(y[-2], s[-2]) / np.inner(y[-2], y[-2])
        z = np.multiply(H_k, q)
        
        for s_i, y_i, a_i in zip(s,y,a):
            b_i = 1.0 * np.inner(y_i, z) / np.inner(y_i, s_i)
            z = z + np.multiply(a_i - b_i, s_i)
        return z

def two_loop_recursion_two(s, y, v):
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = v
        a = []
        m = len(s)-1
        for j in range(m+1):
            rho = 1.0/ np.inner(y[m-j], s[m-j])
            alpha = rho * np.inner(s[m-j], q)
            q = q - np.multiply(alpha, y[m-j])
            a.append( alpha )
            
        H_k = np.inner(y[-2], s[-2]) / np.inner(y[-2], y[-2])
        z = np.multiply(H_k, q)
        
        for i in reversed(range(m+1)):
            rho = 1.0/ np.inner(y[m-i], s[m-i])

            b_i = rho * np.inner(y[m-i], z)
            z = z + np.multiply(a[i] - b_i, s[m-i])
        
        return z


s = [ np.array([1,0,0]) ,np.array([0,1,0]) ,np.array([1,0,1]) ]
y = [ np.array([1,1,0]) ,np.array([0,1,1]) ,np.array([1,2,1]) ]

v = np.array([2,0,0])
print "v", v

H = get_H(s, y, v)
print "H*v:", H.dot(v)
tl = two_loop_recursion(s, y, v)
print "tl1:", tl
tl_z = two_loop_recursion_zip(s, y, v)
print "tlzip:", tl_z
tl2 = two_loop_recursion_two(s, y, v)
print "tl2:", tl2   
