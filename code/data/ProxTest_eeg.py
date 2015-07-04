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

lr = LogisticRegression()
f = lambda x: lr.F(x, X, y)
gf = lambda x: lr.g(x, X, y)

pm.compute_0sr1(f, gf, )