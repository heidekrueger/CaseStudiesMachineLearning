# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:57:01 2015

@author: Fin Bauer
"""

import numpy as np
from SQN.LogisticRegression import LogisticRegression
#import data.ProxMeth as pm
import data.datasets as ds

from PSQN import PSQN

X, y = ds.load_iris()

lr = LogisticRegression()
f = lambda x: lr.F(x, X, y)
gf = lambda w: lr.G(np.array(w.flat), X, y).reshape(w.shape)
xstart=np.zeros(len(X[0]))

sqn = PSQN({'dim':len(X[0])})
sqn.solve(lr.F, lr.G, X, y)

#fval = pm.compute_0sr1(f, gf, xstart)