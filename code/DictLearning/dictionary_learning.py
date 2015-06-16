"""
    @Stan
    Implementation of online dictionary learning algorithms 1 and 2

    DONE :
    - algorithm1
    - structure of algorithm2
    - mini batch
    - add a stopping criterion for while loop in algorithm2
    - test convergence criterion (cf renormalisation)
    - initial dictionary in early steps of algo1
    - Implementation with a .fit method = class ?


    TODO :
    - test it on scikit learn ex.
    - test it on WNVP
    - add a verbose function
    - selection of best regularization parameter l
    - select option : SQN or normal where would be the access point ?
    - write test dictionary learning script
"""


import numpy as np
import math
from time import time


class StochasticDictionaryLearning:
    '''
    This class implements dictionary learning

    Attributes:
    - n_components
    - l, regularization parameter
    - n_iter, int, number of iterations
    - eta, int, mini batch size
    - verbose, int, control verbosity of algorithm


    Methods:
    __init__ :
    fit :
    set_params :
    get_params :

    '''

    def __init__(self, n_components=100, l=0.001, n_iter=30, eta=100, verbose=0):
        self.n_components = n_components
        self.l = l
        self.n_iter = n_iter
        self.eta = eta
        self.verbose = verbose

    def fit(self, X):
        '''
        This methods runs dictionary learning algorithms on data X

        INPUTS:
        - self
        - X : (n_samples, m) array like, data

        OUTPUTS:
        - D : (m, k) array like, learned dictionary
        '''

        D = algorithm1(X,
                       n_components=self.n_components,
                       l=self.l,
                       n_iter=self.n_iter,
                       eta=self.eta,
                       verbose=self.verbose)
        return D


def load_data():
    '''
    code from sklearn application of dictionary learning

    RETURNS:
    - data: n_patches, dim_patch array like. collection of patches
    '''
    from sklearn.feature_extraction.image import extract_patches_2d
    from scipy.misc import lena

    # Load Lena image and extract patches

    lena = lena() / 256.0

    # downsample for higher speed
    lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
    lena /= 4.0
    height, width = lena.shape

    # Distort the right half of the image
    print('Distorting image...')
    distorted = lena.copy()
    distorted[:, height // 2:] += 0.075 * np.random.randn(width, height // 2)

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    patch_size = (7, 7)
    data = extract_patches_2d(distorted[:, :height // 2], patch_size)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))

    return data


def algorithm1(x, n_components=100, l=0.01, n_iter=30, eta=40, verbose=0):
    '''
    Online dictionary learning algorithm

    INPUTS:
    - x : (n_samples, m) array like, data
    - l, regularization parameter
    - n_components, number of components of dictionary
    - n_iter, int, number of iterations
    - eta, int, mini batch size
    - verbose, int, control verbosity of algorithm

    OUTPUTS:
    - D : (m, k) array like, learned dictionary
    '''

    n_s = len(x[:, 0])  # number of samples in x

    D = np.random.rand(len(x[0, :]), n_components)  # initial dictionary
    print D.shape

    # Dimensions of dicitonary
    [m, k] = D.shape

    # 1: initialization
    A = np.zeros((k, k))
    B = np.zeros((m, k))

    # 2: Loop
    for t in range(1, n_iter + 1):

        # 3: Draw xj from x
        j = np.random.randint(0, n_s, eta)
        xt = x[j, :]
        xt = np.asmatrix(xt).T

        # 4: Sparse coding with LARS
        from sklearn.linear_model import LassoLars
        lars = LassoLars(alpha=l)

        lars.fit(D, xt)
        alpha = lars.coef_
        alpha = (np.asmatrix(alpha)).T

        # computing coefficient beta for step 5/6
        if t < eta:
            theta = float(t * eta)
        else:
            theta = math.pow(eta, 2) + t - eta

        beta = (theta + 1 - eta) / (theta + 1)

        # 5: Update A
        a = np.zeros((k, k))
        for i in range(0, eta):
            a = a + (alpha[:, i]).dot(alpha[:, i].T)
        A = beta * A + a

        # 6: Update B
        b = np.zeros((m, k))
        for i in range(0, eta):
            b = b + xt[:, i].dot(alpha[:, i].T)
        B = beta * B + b

        # Compute new dictionary update
        D = algorithm2(D, A, B)

    # 9 : Return learned dictionary
    return D


def algorithm2(D, A, B, c_max=15, eps=0.00001):
    '''
    Dictionary update

    INPUTS:
    - D, (m, k), input dictionary
    - A, (k, k)
    - B, (m, k)
    - c_max, int, max number of iterations
    - eps, float, stopping criterion

    OUTPUT:
    - D, updated dictionary
    '''

    m = len(D[:, 0])
    k = len(D[0, :])

    c = 0  # counter
    cv = False  # convergence or stop indicator

    # 2: loop to update each column
    while cv is not True:

        # keep a trace of previous dictionary
        D_old = np.zeros((m, k))
        for i in range(0, m):
            for j in range(0, k):
                D_old[i, j] = D[i, j]

        for j in range(0, k):

            # 3: Update the j-th column of d
            u = (1 / A[j, j]) * (B[:, j] - D.dot(A[:, j]))
            u = u + np.asmatrix(D[:, j]).T

            # renormalisation
            renorm = max(np.linalg.norm(u), 1)
            u = np.divide(u, renorm)

            for p in range(0, m):
                D[p, j] = u[p]

        # counter update
        c = c + 1

        # compute differences between two updates
        grad = D - D_old
        crit = np.linalg.norm(grad)

        # check convergence
        if crit < eps:
            cv = True
        if c > c_max:
            cv = True

    if c == c_max:
        print "Dictionary Updating Algo reached max number of iterations"
        print "Consider higher max number of interations"

    # 6: Return updated dictionary
    return D


if __name__ == "__main__":
    sdl = StochasticDictionaryLearning()
    X = load_data()

    sdl.fit(X)

    # algorithm1(x)
