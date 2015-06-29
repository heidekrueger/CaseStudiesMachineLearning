"""
This file contains a function to load eeg data.
"""

import numpy as np


def load_eeg():
    '''
    This function loads .npy files X and y which correspond to eeg recordings
    and labels

    NB :
    - Pay attention to label file dimension
    - You may want to convert the label file into a vector
    '''

    data_name = 'data.npy'
    label_name = 'label.npy'

    X = np.load(data_name)
    y = np.load(label_name)

    print "Data dim : ", X.shape
    print "Label dim : ", y.shape

    return X, y
