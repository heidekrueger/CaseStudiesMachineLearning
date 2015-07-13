# -*- coding: utf-8 -*-
"""
Created on Tue Jul 7 10:06:07 2015

@author: Fin Bauer
"""

import numpy as np
import ProxMeth as pm

def dl_prox_meth(D, x, l_reg, max_iter=1e5):
    """
    Computes optimal point alpha for Lasso:
    1/(2*nsamples) ||D*alpha - x||² + l_reg*||alpha||_1

    Arguments
    ---------
    D : Dictionary
    x : target
    l_reg : regularization factor

    Returns
    -------
    alpha

    """

    m, n = np.shape(D)
    alpha0 = np.zeros(shape=(n,1))
    
    def dl_lasso_f(alpha):

        return 0

    def dl_lasso_gf(alpha):

        D_sq = np.dot(D.T, D)

        return  1 / m * np.dot(D_sq, alpha) - np.dot(D.T, x)

    alpha, _, _, _ = pm.compute_0sr1(dl_lasso_f, dl_lasso_gf, alpha0, timing=1,
                                     l_reg=l_reg, max_iter=max_iter,
                                     epsilon=1e-6)

    return alpha