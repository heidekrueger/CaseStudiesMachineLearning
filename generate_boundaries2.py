"""
    @author : stan

    This script is made to generate sets of points from different
    classes and plot them and boundaries between
"""

import numpy as np
import matplotlib.pyplot as plt

n_set = 4  # number of sets / classes
n_pts = 200  # number of points per class
colormap = plt.cm.winter

means = [[2, -2], [-5, 5], [3, 5], [-4, -4]]
cov0 = [[1, 0], [0, 1]]
cov1 = [[5, 3], [3, 5]]
cov2 = [[0, 1], [1, 0]]
cov3 = [[2, 1], [1, 2]]

cov = [cov0, cov1, cov2, cov3]
colors = []
for i in range(1,5):
    colors.append(colormap(i/4.0))

x = []
y = []
X = []

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

# plt.show()

X = np.zeros((n_pts * n_set, 2))
Y = np.zeros((n_pts * n_set))
for i in range(0, n_set):
    x0 = x[i]
    y0 = y[i]
    for j in range(0, n_pts):
        X[j + i * n_pts, :] = [x0[j], y0[j]]
        Y[j + i * n_pts] = i

num_subsamples = int(0.1 * n_pts)

sampled_indexes = [np.random.randint(0, X.shape[0])
                   for i in range(num_subsamples)]

X_sub = np.asarray([X[i] for i in sampled_indexes])
Y_sub = np.asarray([Y[i] for i in sampled_indexes])

h = .02  # step size in the mesh
from sklearn import svm

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 10  # SVM regularization parameter
print("1")
svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
print("2")

svc_sub = svm.SVC(kernel='poly', degree=3, C=C).fit(X_sub, Y_sub)
print("3")
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Boundaries found by SVM: Whole data set',
          'Boundaries found by SVM: Subsampled data set',
          'Comparison']

X_plot = [X, X_sub, X]
Y_plot = [Y, Y_sub, Y]
for i, clf in enumerate((svc, svc_sub, svc_sub)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    #plt.subplot(1, 1, i + 1)
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    fig = plt.figure()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.8)
    print( colormap 	)
    # Plot also the training points
    plt.scatter(X_plot[i][:, 0], X_plot[i][:, 1], c=Y_plot[i], cmap=colormap)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
		
    
    #ax = fig.add_subplot(111)
    #ax.imshow(data,interpolation='none')

    fig.savefig('plot %d.eps' %i)

plt.show()
