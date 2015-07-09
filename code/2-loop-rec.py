import numpy as np
import itertools


def get_H(s,y):
        
        I = np.identity(len(s[0]))
        H = np.dot((np.inner(s[-1], y[-1]) / np.inner(y[-1],
                   y[-1])), I)

        for (s_j, y_j) in itertools.izip(s, y):
            rho = 1.0/np.inner(y_j, s_j)
            H = (I - rho * np.outer(s_j, y_j)).dot(H).dot(I - rho * np.outer(y_j, s_j))
            H += rho * np.outer(s_j, s_j)
            

        return H

def two_loop_recursion(s, y, v):
        # H = (s_t^T y_t^T)/||y_t||^2 * I

        q = v
        rho = 1.0/ np.inner(y[-1], s[-1])
        a = []

        for j in range(len(s)):
            a.append( rho * np.inner(s[j], q))
            q = q - np.multiply(a[-1], y[j])

        H_k = np.inner(y[-2], s[-2]) / np.inner(y[-2], y[-2])
        z = np.multiply(H_k, q)

        for j in reversed(range(len(s))):
            b_j = rho * np.inner(y[j], z)
            q = q - np.multiply(a[j] - b_j, s[j])

        return z


s = [ np.array([1,0,0]),np.array([0,1,0]),np.array([1,0,1]) ]
y = [ np.array([1,1,0]),np.array([0,1,1]),np.array([1,2,1]) ]

v = np.array([2,5,4])

H = get_H(s, y)
print H
print H.dot(v)

print two_loop_recursion(s, y, v)
