# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:35:10 2015

@author: Fin Bauer
"""


def prox_comparison():
    
    import matplotlib.pyplot as plt
    import Prox_LogReg as lr
    import StochProxGrad as spg
    import StochProxMeth as spm
    from matplotlib2tikz import save as tikz_save
    
    fval_spm, x1 = spm.compute_0sr1(lr.F, lr.G, x0, X, y, l_reg = l, tau = 0.1, batch_size = batch_size)
    fval_spg, x2 = spg.proximal_gradient(lr.F, lr.G, x0, 0.1, X, y, l_reg = l, batch_size = batch_size)
    fval_spm.insert(0, lr.F(x0, X, y))
    fval_spg.insert(0, lr.F(x0, X, y))
    
    line1, = plt.plot(range(len(fval_spm)), fval_spm, 'r', label = 'S0SR1', lw = 2)
    line2, = plt.plot(range(len(fval_spg)), fval_spg, 'b', label = 'SPG', lw = 2)
    
    #plt.xlim([0, 55])
    #plt.ylim([1e1, 1e13])
    plt.ylabel('Function Value')
    plt.xlabel('Number of Iterations')
    plt.legend(handles = [line1, line2])
    #tikz_save( 'myfile2.tikz' );

    return x1, x2

def predict(w, X):
    """
    Prediction
    """
    import Prox_LogReg as lr
    m, n = np.shape(X)
    pred = np.zeros(m)
    for i in range(m):
        pred[i] = lr.sigmoid(w, X[i].reshape(n, 1))
        
    return pred

if __name__ == "__main__":
    
    import numpy as np
    
    X = np.random.normal(size = (1000,50))
    y = np.random.randint(0, 2, 1000).reshape(1000, 1)
    batch_size = 10
    x0 = np.random.normal(size = (50, 1))
    l = 0.1
    
    x1, x2 = prox_comparison()
    pred1 = predict(x1, X)
    pred2 = predict(x2, X)