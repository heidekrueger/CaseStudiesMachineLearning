# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:57:01 2015

@author: Fin Bauer
"""

import numpy as np
from SQN.LogisticRegression import LogisticRegression

from PSQN import PSQN
import ProximalMethod.ProxMeth as pm
import data.datasets as ds

X, y = ds.load_eeg()

lr = LogisticRegression()
print("X, y loaded, initializing F")
f = lambda x: lr.F(np.array(x.flat), X, y)
print("done. initializing gf")
gf = lambda w: lr.G(np.array(w.flat), X, y).reshape(w.shape)
print("done.")

xstart=np.zeros(len(X[0]))

sqn = PSQN({'dim':len(X[0])})
sqn.solve(lr.F, lr.G, X, y)

fval = pm.compute_0sr1(f, gf, xstart)
