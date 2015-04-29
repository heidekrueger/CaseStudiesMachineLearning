"""
    @author : stan

    This script is made to generate sets of points from different
    classes and plot them and boundaries between
"""

import numpy as np
import matplotlib.pyplot as plt

n_set = 4  # number of sets / classes
n_pts = 500  # number of points per class

means = [[0, 0], [-2, 3], [3, 2], [-1, 1]]
cov0 = [[1, 0], [0, 5]]
cov1 = [[1, 1], [1, 1]]
cov2 = [[0, 1], [1, 0]]
cov3 = [[2, 1], [1, 2]]

cov = [cov0, cov1, cov2, cov3]

x = []
y = []
# x, y = np.random.multivariate_normal(mean, cov, 500).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()
