# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:57:01 2015

@author: Fin Bauer
"""

import numpy as np
from SQN.LogisticRegression import LogisticRegression
import data.ProxMeth as pm
import data.datasets as ds

X, y = ds.load_eeg()
y = y.reshape(len(y), 1)

lr = LogisticRegression()
print "X, y loaded, initializing F"
f = lambda x: lr.F(x, X, y)
print "done. initializing gf"
gf = lambda w: lr.G(np.array(w.flat), X, y).reshape(w.shape)
print "done."
# x0 finden etc. hofffen dass es läuft
xstart=np.zeros(600)

fval = pm.compute_0sr1(f, gf, xstart)