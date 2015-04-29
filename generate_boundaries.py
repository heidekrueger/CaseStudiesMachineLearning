"""
    @author : stan

    This script is made to generate sets of points from different
    classes and plot them and boundaries between
"""

import numpy as np
import matplotlib.pyplot as plt

n_set = 4  # number of sets / classes
n_pts = 1000  # number of points per class

means = [[2, -2], [-5, 5], [3, 5], [-4, -4]]
cov0 = [[1, 0], [0, 1]]
cov1 = [[5, 3], [3, 5]]
cov2 = [[0, 1], [1, 0]]
cov3 = [[2, 1], [1, 2]]

cov = [cov0, cov1, cov2, cov3]
colors = ['r', 'b', 'y', 'g']

x = []
y = []
<<<<<<< HEAD

for i in range(0, 4):
    m0 = means[i]
    cov0 = cov[i]
    x0, y0 = np.random.multivariate_normal(m0, cov0, n_pts).T
    x.append(x0)
    y.append(y0)

for i in range(0, 4):
    x0 = x[i]
    y0 = y[i]
    plt.scatter(x0, y0, c=colors[i], edgecolors='none')

plt.show()

=======
x, y = np.random.multivariate_normal(means[0], cov0, 500).T


from matplotlib.markers import MarkerStyle

marker = MarkerStyle(marker=None, fillstyle=u'full')
Bases: object

MarkerStyle

Parameters:	
marker : string or array_like, optional, default: None

See the descriptions of possible markers in the module docstring.

fillstyle : string, optional, default: ‘full’

‘full’, ‘left”, ‘right’, ‘bottom’, ‘top’, ‘none’

Attributes

markers	(list of known markes)
fillstyles	(list of known fillstyles)
filled_markers	(list of known filled markers.)
filled_markers = (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
fillstyles = (u'full', u'left', u'right', u'bottom', u'top', u'none')



plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
>>>>>>> 2a13995c3fe43ef49b8d5be45ca578c6d61c8652
