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
