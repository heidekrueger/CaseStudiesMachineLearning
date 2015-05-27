"""
    @Stan
    Implementation of online dictionary learning algorithms 1 and 2

    DONE :
    - algorithm1
    - structure of algorithm2

    TODO :
    add a stopping criterion for while loop in algorithm2
"""


import numpy as np
from time import time


# Global variables
m = 49
k = 20
l = 0.00001
d0 = np.random.rand(m, k)
t = 3


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


def algorithm1(x, l, D, t):
    '''
    Online dictionary learning algorithm

    INPUTS:
    - x : (n_samples, m) array like, data
    - l, regularization parameter
    - d0, (m, k) initial dictionary
    - t, int, number of iterations
    '''
    n_s = len(x[:, 0])

    # 1: initialization :
    # A : (k, k) zero matrix
    # B : (m, k) zero matrix
    A = np.zeros((k, k))
    B = np.zeros((m, k))
    print A.shape
    print B.shape

    # 2: Loop
    for i in range(1, t):
        # 3: Draw xj from x
        j = np.random.randint(0, n_s)
        xj = x[j, :]

        # 4: Sparse coding with LARS
        from sklearn.linear_model import LassoLars
        lars = LassoLars(alpha=l)

        lars.fit(D, xj)
        alpha = lars.coef_
        alpha = (np.asmatrix(alpha)).T

        # 5: Update A
        A = A + (alpha).dot(alpha.T)

        # 6: Update B
        xj = (np.asmatrix(xj)).T
        B = B + xj.dot(alpha.T)

        D = algorithm2(D, A, B)

    # 9 : Return learned dictionary
    return D


def algorithm2(D, A, B):
    '''
    Dictionary update

    INPUTS:
    - D, (m, k), input dictionary
    - A, (k, k)
    - B, (m, k)

    OUTPUT:
    - D, updated dictionary
    '''

    # counter to simulation a stopping criterion
    c = 0
    c_max = 10

    # Loop until convergence => What kind of cv ?
    while c < c_max:
        c = c + 1
        for j in range(0, k):

            # 3: Update the j-th column of d
            u = (1 / A[j, j]) * (B[:, j] - D.dot(A[:, j]))
            u = u + np.asmatrix(D[:, j]).T

            # renormalisation
            renorm = max(np.linalg.norm(u), 1)
            for p in range(0, m):
                D[p, j] = u[p] / renorm

    return D


if __name__ == "__main__":
    x = load_data()

    algorithm1(x, l, d0, t)
