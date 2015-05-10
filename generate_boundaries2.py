"""
    @author : stan

    This script is made to generate sets of points from different
    classes and plot them and boundaries between
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib as mpl
from sklearn import svm

n_set = 4  # number of sets / classes
classes = range(n_set)
n_pts = 30  # number of points per class
num_subsamples = int(5)
mywhite = '#F1F0F0'
myblue = '#03508D'
myred = '#D22727'
myblack = '#0E070A'
mygray = '#727273'

mpl.rc('axes',edgecolor=mywhite) #set default axis color

# colormap = plt.cm.coolwarm # the python colormap that defines the colors of the plot
colormap = pltcol.LinearSegmentedColormap.from_list('mycmap', [(0 / 3.0, myblue),
                                                    (1 / 3.0, mywhite),
                                                    (2 / 3.0, mygray),
                                                    (3/3.0, myred)]
                                        )

symSize = 350 # the size of data points

means = [[2, -2], [-5, 5], [3, 5], [-4, -4]]
cov0 = [[1, 0], [0, 1]]
cov1 = [[5, 3], [3, 5]]
cov2 = [[0, 1], [1, 0]]
cov3 = [[2, 1], [1, 2]]

cov = [cov0, cov1, cov2, cov3]              # diamond       #hearts
markers = [r'$\clubsuit$', r'$\spadesuit$', ur'$\u2666$', ur'$\u2665$']
colors = [myblue,mywhite,mygray,myred]
edgecolors = [mywhite,'none',mywhite,mywhite]

x = []
y = []
X = []

fig = plt.figure(facecolor=None)

for i in range(0, 4):
    m0 = means[i]
    cov0 = cov[i]
    x0, y0 = np.random.multivariate_normal(m0, cov0, n_pts).T
    x.append(x0)
    y.append(y0)

for i in range(0, 4):
    x0 = x[i]
    y0 = y[i]
    plt.scatter(x0, y0, c=colors[i], s=symSize, edgecolor=edgecolors[i], marker=markers[i])

plt.xlabel('Hauptkomponente 1',color=mywhite)
plt.ylabel('Hauptkomponente 2', color = mywhite)
plt.xticks(())
plt.yticks(())
# add unknown data point
plt.scatter(-3,2.5,c=myblack,s=symSize*2,marker = r'$?$',edgecolor='none')
# plt.scatter(-3,2,c='none',s=symSize+8,marker = 'o',edgecolor='y') # draw a circle around the '?'

fig.savefig('plot %d.eps' % 0, transparent=True)
# plt.show()

X = np.zeros((n_pts * n_set, 2))
Y = np.zeros((n_pts * n_set))
for i in classes: 
    x0 = x[i]
    y0 = y[i]
    for j in range(0, n_pts):
        X[j + i * n_pts, :] = [x0[j], y0[j]]
        Y[j + i * n_pts] = i



sampled_indexes = [np.random.randint(0, X.shape[0])
                   for i in range(num_subsamples)]

X_sub = np.asarray([X[i] for i in sampled_indexes])
Y_sub = np.asarray([Y[i] for i in sampled_indexes])

h = .02  # step size in the mesh


# we create an instance of SVM and fit our data. We do not scale our
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
    # plt.subplot(1, 1, i + 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    fig = plt.figure()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.5)
    print(colormap 	)
    # Plot also the training points
    for cl in classes:
        plt.scatter(X_plot[i][cl*n_pts:cl*n_pts+n_pts, 0],
                    X_plot[i][cl*n_pts:cl*n_pts+n_pts, 1],
                    c=colors[cl],
                    s=symSize,
                    marker=markers[cl]
                    )
    plt.scatter(-3,2.5,c=myblack,s=symSize*2,marker = r'$!$', edgecolor='none')
    plt.xlabel('Hauptkomponente 1',color=mywhite)
    plt.ylabel('Hauptkomponente 2', color = mywhite)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks((),color = mywhite)
    plt.yticks((), color = mywhite)
    # plt.title(titles[i], color = mywhite)

    # ax = fig.add_subplot(111)
    # ax.imshow(data,interpolation='none')

    fig.savefig('plot %d.eps' % (i+1), transparent=True)

plt.show()
