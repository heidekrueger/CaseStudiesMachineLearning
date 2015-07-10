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

from sqndict import SqnDictionaryLearning
from joblib import Parallel, delayed


def inner_rec(lena, data, intercept, d, algo='omp', nzc=None):
    '''
    INPUTS:
    - lena
    - data
    - intercept
    - d : dictionary
    - algo : reconstruction algorithm to use
    - nzc : number of non zero coefficients

    OUTPUTS:
    - reconstructed photograph
    '''
    # copy the photo
    rec = lena.copy()

    # encode noisy patches with learnt dictionary
    code = sparse_encode(data,
                         d.T, gram=None,
                         cov=None,
                         algorithm=algo,
                         n_nonzero_coefs=nzc,
                         alpha=None,
                         copy_cov=True,
                         init=None,
                         max_iter=1000,
                         n_jobs=1)

    # dimension determination
    patch_size = (7, 7)
    height, width = lena.shape

    # patch reconstruction
    patches = np.dot(code, d.T)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)

    # reconstruct noisy image
    rec[:, height // 2:] = reconstruct_from_patches_2d(patches,
                                                       (width, height // 2))

    # return reconstructed image
    return rec


def outer_reconstruction(influence, lena,
                         pi='n_components', algo='omp', nzc=None, n_jobs=1):
    '''
    INPUTS:
    - influence : l_params, l_dict
    - lena : photograph to reconstruct
    - pi : parameters which is investigated
    - algo : reconstruction algo to use
    - nzc : number of non zero coefficients
    - n_jobs : number of jobs to run in parallel

    OUTPUTS:
    - Plot reconstructed photographs
    '''

    # data preprocessing
    data, lena, distorted = preprocess_data(lena)

    # post process data
    data, intercept = postprocess_data(lena)

    l_params = influence[0]
    l_dict = influence[1]

    # parallel loop for parallelization
    rec = Parallel(n_jobs=n_jobs)(delayed(inner_rec)(lena,
                                                     data,
                                                     intercept,
                                                     d,
                                                     algo=algo,
                                                     nzc=nzc)
                                  for d in l_dict)
    print len(rec)

    # plot reconstructed images
    for i in range(0, len(rec)):
        p = l_params[i]
        show_with_diff(rec[i], lena, pi + ' : ' + str(p[pi]))

    # plot distorded image
    show_with_diff(distorted, lena, 'Distorted image')

if __name__ == '__main__':
    # loads lena
    from scipy.misc import lena

    # loads dictionaries
    influence = np.load('influence_n_iter.npy')

    outer_reconstruction(influence,
                         lena,
                         pi='n_iter',
                         algo='lars',
                         nzc=10,
                         n_jobs=5)

    # plot dictionaries
    plt.show()
