# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:57:01 2015

@author: Fin Bauer
"""

import numpy as np
from LogisticRegression import LogisticRegression
import ProxMeth as pm
import datasets as ds

X, y = ds.load_eeg()
y = y.reshape(len(y), 1)

lr = LogisticRegression()
f = lambda x: lr.F(x, X, y)
gf = lambda w: sum([lr.g(w, x, z) for x,z in zip(X,y)])/len(y)

# x0 finden etc. hofffen dass es läuft
xstart=np.zeros(600)

fval = pm.compute_0sr1(f, gf, xstart)