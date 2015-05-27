"""
    @Stan
    Implementation of online dictionary learning algo
"""


import numpy as np
from time import time


# Global variables
m = 20
k = 10
lamb = 1.0
D_0 = np.random.rand(m, k)
T = 100


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


# def generate(n):
#     '''
#     This functions generate iid rv from R^m
#     '''
#     mu = np.random.rand(m)
#     S = np.random.rand(m, m)
#     S = 1 + (S + np.transpose(S)) / 2

#     X = np.random.multivariate_normal(mu, S, n)

#     return X


# def online_dict_learning()


# def lars_solver(x, D, lamb):
# 	'''
# 	This function solves the min problem (8)
# 	'''

# def dict_uptdater(D, A, B, X, lamb):
# 	'''
#     This function compute the new update of D cf algo 2
#     what is a warm restart ?
#     '''
#     # while loop : criterion of convergence ?
#     for j in range(0, k):
#         u = (1 / A[j, j]) * (b[:, j] - np.dot(D, a[:, j]) + D[:, j])

#         # renormalisation
#         renorm = max(np.linalg.norm(u), 1)
#         d[:, j] = u / renorm

# 	return D

if __name__ == "__main__":
    data = load_data()
