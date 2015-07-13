import numpy as np
import random as rd
import math


def sample_batch_SQL(w, X, z=None, b=None, r=None, debug=False):

    assert b is not None or r is not None, \
        "Choose either absolute or relative sample size!"

    assert (b is not None) != (r is not None), \
        "Choose only one: Absolute or relative sample size!"

    N = len(X)
    if b is not None:
        nSamples = b
    else:
        nSamples = r*N
    if nSamples > N:
        if debug:
            print("Batch size larger than N, using whole dataset")
        nSamples = N
    #
    # Draw from uniform distribution
    #
    random_indices = rd.sample(range(N), int(nSamples))

    if debug:
        print("random indices", random_indices)

    TABLE_NAME = 'table'
    KEY_FIELD_NAME = 'rowid'
    query_str = 'SELECT t.* FROM ' + TABLE_NAME \
                + ' t WHERE t.' + KEY_FIELD_NAME \
                + ' IN (' + ','.join([str(i) for i in random_indices]) + ');'

    if debug:
        print('Opening connection...')

    raise NotImplementedError

    X_S = np.asarray([X[i] for i in random_indices])
    z_S = np.asarray([z[i] for i in random_indices]) if z is not None else None

    if debug:
        print(X_S, z_S)

    if z is None or len(z) == 0:
        return X_S, None

    else:
        return X_S, z_S


def load_HIGGS_feature(ID):
    feature_file = open("../datasets/HIGGS/%d" % ID)
    line = feature_file.readline()
    entries = line.split(",")
    y = int(float(entries[0]))
    entries[0] = 1.
    return np.array([float(e) for e in entries]), y


def load_HIGGS_features(id_list):
    X, z = [], []
    for ID in id_list:
        x, y = load_HIGGS_feature(ID+1)
        X.append(x)
        z.append(y)
    return X, z


def sample_batch(w, X, z=None, b=None, r=None, debug=False):
    """

    #TODO: Documentation Outdated!!!
    returns a uniformly distr. subset of [N] as a list

    Parameters:
        N: Size of the original set
        b: parameter for subsample size (e.g. b=.1)
    """

    assert b is not None or r is not None, \
        "Choose either absolute or relative sample size!"
    assert (b is not None) != (r is not None), \
        "Choose only one: Absolute or relative sample size!"

    N = len(X)
    if b is not None:
        nSamples = b
    else:
        nSamples = r*N
    if nSamples > N:
        if debug:
            print("Batch size larger than N, using whole dataset")
        nSamples = N
    #
    # Draw from uniform distribution
    #
    random_indices = rd.sample(range(N), int(nSamples))
    if debug:
        print("random indices", random_indices)

    X_S = np.asarray([X[i] for i in random_indices])
    z_S = np.asarray([z[i] for i in random_indices]) if z is not None else None

    if debug:
        print(X_S, z_S)

    if z is None or len(z) == 0:
        return X_S, None
    else:
        return X_S, z_S


def stochastic_gradient(g, w, X=None, z=None):
    """
    Calculates Stochastic gradient of F at w as per formula (1.4)
    """
    if X is not None:
        nSamples = len(X)
        nFeatures = len(X[0])

    # print(nSamples)
    # print(X[0].shape, w.shape)
    if X is None:
        return g(w)
    if z is None:
        return np.array(sum([g(w, X[i]) for i in range(nSamples)]))
    else:
        assert len(X) == len(z), "Error: Dimensions must match"
        # print(" one gradient:" , g(w,X[0],z[0]))
        return sum([g(w, X[i], z[i]) for i in range(nSamples)])


def armijo_rule(f, g, x, s, start=1.0, beta=.5, gamma=1e-4):
    """
    Determines the armijo-rule step size alpha for approximating
    line search min f(x+omega*s)

    Parameters:
        f: objective function
        g: gradient
        x:= x_k
        s:= x_k search direction
        beta, gamma: parameters of rule
    """
    candidate = start
    #print("armijo")
    #print(f(x + np.multiply(candidate, s)) )
    #print("fa", f(x))
    #print(candidate * gamma * np.dot( g(x).T, s))
    #print(s)
    #print("---")
    while (f(x + np.multiply(candidate, s)) - f(x) > candidate * gamma * np.dot( g(x).T, s)) and candidate > 1e-4:
    
    #   print("armijo")
    #   print(f(x + np.multiply(candidate, s)) - f(x))
    #   print(candidate * gamma * np.dot( g(x).T, s))
        
        candidate *= beta
    return candidate


'''
    general tools
'''


def iter_to_array(iterator):
    return np.array([i for i in iterator])


def set_iter_values(iterator, w):
    for i in range(len(w)):
        iterator[i] = w[i]

from scipy.signal import welch
from statsmodels.tsa.stattools import acovf
def stationarity_convergence(phi):
    
        m = 20 # burn in 
        n = len(phi) - m
        if n <= 0:
                return 0
        na = max(1,int(0.1*n))
        nb = max(1,int(0.5*n))
        
        phi_a = phi[(m+n-na):(m+n)]
        phi_b= phi[(m):(m+nb)]
        phi_b= phi[(m+n-nb):(m+n)]
        
        phi_b_bar = sum(phi_b)/nb
        phi_a_bar = sum(phi_a)/na
        if len(phi_a) <= 1 or len(phi_b) <= 1:
                return 1
        
        v_a = acovf(phi_a)
        v_b = acovf(phi_b)
        
        # n gets large and na/n and nb/n stay fixed
        z_g = (phi_a_bar - phi_b_bar)/np.sqrt( v_a[0] + v_b[0] )
        return z_g

from statsmodels.stats.diagnostic import lillifors
def test_normality(f, level = 0.01, burn_in = 200):
        if len(f) <= burn_in + 1:
                return False
        return lillifors(f)[1] > level
    
def test_stationarity(f_evals):
    """
    Tests for stationarity of the input sequence using the sample
    autocorrelation function

    TODO: seems to reject samples to often (~3/20 times for iid std
    normal series!)
    """
    n = len(f_evals)
    # number of points that will be considered for stationarity:
    m = 50
    M = 60
    if n < m:
        print("too small")
        return False
    else:
        m = max(m, 0.1*len(f_evals))
        m = min(m, M)
        # consider only the last m points
        f_evals = f_evals[n-m:]

        fbar = 1/float(m)*sum(f_evals)

        autocorrs = []
        for h in range(int(0.4*m)):
            gamma = 0
            for j in range(m-h):
                gamma += (f_evals[j+h]-fbar)*(f_evals[j]-fbar)
            gamma /= m
            autocorrs.append(gamma)

        # Find entries in 95%-Confidence interval
        in_confidence_interval = filter(lambda x: x > -1.69/math.sqrt(m) and x < 1.69/math.sqrt(m), autocorrs)
        print(len(in_confidence_interval))
        if len(in_confidence_interval)/0.4/m < .90:
            return False
        else:
            print("Stationarity reached.")
            return True
