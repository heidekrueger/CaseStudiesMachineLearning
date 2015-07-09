"""
@author : Stan
This script studies the influence of the reconstruction algo on performance
"""

from dictionary_learning_test import postprocess_data, preprocess_data
from dictionary_learning_test import show_with_diff, plot_dictionary
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition.dict_learning import sparse_encode
import numpy as np
from time import time
import matplotlib.pyplot as plt


def influence(D, algo='omp', n_nonzero_coefs=None):
    '''
    reconstructs the distorted image

    INPUTS:
    - D : dictionary to use
    - algo : reconstruction algorithm : 'omp', 'lars'
    - n_nonzero_coefs : int, necessary to control the accuracy of the
    reconstruction

    OUTPUS:
    - picture of the distorted image
    - picture of the reconsted image
    '''

    from scipy.misc import lena

    data, lena, distorted = preprocess_data(lena)

    # post process data
    data, intercept = postprocess_data(lena)

    t0 = time()
    title = algo + 'with ' + str(n_nonzero_coefs) + ' n atoms'
    reconstructions = lena.copy()

    # encode noisy patches with learnt dictionary
    code = sparse_encode(data, D.T, gram=None,
                         cov=None, algorithm=algo,
                         n_nonzero_coefs=n_nonzero_coefs, alpha=None,
                         copy_cov=True, init=None,
                         max_iter=1000, n_jobs=1)

    patch_size = (7, 7)
    height, width = lena.shape

    patches = np.dot(code, D.T)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)

    # reconstruct noisy image
    reconstructions[:, height // 2:] = reconstruct_from_patches_2d(
        patches, (width, height // 2))

    dt = time() - t0
    print('done in %.2fs.' % dt)

    # show the difference
    show_with_diff(distorted, lena, 'Distorted image')
    show_with_diff(reconstructions, lena,
                   title + ' (time: %.1fs)' % dt)

if __name__ == '__main__':

    # loads dictionary
    D = np.load('dictionary.npy')

    # plots dictionary for fun
    plot_dictionary(D)

    # creates a list of reconstruction algo and parameters
    l_algos = []
    for i in range(1, 10):
        l_algos.append({'algo': 'lars',
                        'n_nonzero_coefs': i})

    # performs reconstruction for each set of parameters
    for a in l_algos:
        influence(D, algo=a['algo'], n_nonzero_coefs=a['n_nonzero_coefs'])
    plt.show()
