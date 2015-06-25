"""
    @Stan
    StochasticDictionaryLearningTest

    This script is made for comparing our implementation of DictionaryLearning
    algorithm and scikit learn's one.

    TODO:
    - think about class organisation
    - fix coding/writting conventions problem
    - create a function to run scikit's algo and store results
    - create a function to run our algo and store results
    - create a function to display results for both algorithm
        => display pictures
        => display running time, number of iterations...
        => display performances (how to measure it ?)
    - reproduce scikit's results with our implementation
    - try to do better with incorporating SQN, and proximal methods

    DONE:
    - load data
    - reconstruct data
    - run algorithm

"""
from dictionary_learning import StochasticDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
from time import time
import matplotlib.pyplot as plt


class StochasticDictionaryLearningTest(StochasticDictionaryLearning):
    '''
    Class to test dictionary learning algorithm and compare it to sklearn's one

    Attributes:
    - ours : our algorithm
    - sklearn : sklearn algorithm
    - n_components
    - option, select SQN method or normal method
    - l, regularization parameter
    - n_iter, int, number of iterations
    - eta, int, mini batch size
    - verbose, int, control verbosity of algorithm

    methods:
    - __init__
    - pipeline : sklearn's example of dictionary learning
    - print_attributes
    - plot_results
    '''

    def __init__(self):
        StochasticDictionaryLearning.__init__(self)

        def test_dict(self, X):
            '''
            '''


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


def load_data_Fin_Jakob():
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
    distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    patch_size = (7, 7)

    data_left = extract_patches_2d(distorted[:, :width // 2], patch_size)
    data_left = data_left.reshape(data_left.shape[0], -1)
    data_left -= np.mean(data_left, axis=0)
    data_left /= np.std(data_left, axis=0)

    data_right = extract_patches_2d(distorted[:, width // 2:], patch_size)
    data_right = data_right.reshape(data_right.shape[0], -1)
    center = np.mean(data_right, axis=0)
    data_right -= center
    print('done in %.2fs.' % (time() - t0))

    return distorted, data_left, data_right, center


def reconstruct_data(D, data, dist, center):
    """
    reconstruct right side from learned dictionary

    INPUTS:
    - D : learned dictionary
    - data: patches from right side
    - dist: array of full distorted image
    - center: mean of data to reverse centralization

    OUTPUTS:
    - recon: array of reconstructed image
    """

    t0 = time()
    recon = dist.copy()
    patch_size = (7, 7)
    lars = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
    lars.fit(D, data.T)
    code = lars.coef_
    patches = np.dot(code, D.T)
    patches += center
    patches = patches.reshape(len(data), *patch_size)
    recon[:, dist.shape[1] // 2:] = reconstruct_from_patches_2d(patches, (dist.shape[0], dist.shape[1] // 2))
    print('done in %.2fs.' % (time() - t0))

    return recon


if __name__ == '__main__':
    '''
    This stuffs should be embedded in a function or class method
    '''
    sdl = StochasticDictionaryLearning()

    # Loading data
    distorted, data_left, data_right, center = load_data_Fin_Jakob()

    # Learning dictionary
    print("Fitting dictionary...")
    D = sdl.fit(data_left)
    print("Dictionary fitted.")

    # Reconstructing/Denoising image
    print("Reconstructing image...")
    recon = reconstruct_data(D, data_right, distorted, center)
    print("Image reconstructed.")

    plt.gray()
    plt.imshow(distorted)
    plt.show()
    plt.imshow(recon)
    plt.show
