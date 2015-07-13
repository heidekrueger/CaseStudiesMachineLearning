import numpy as np
import math
from SGD import SQN


class LogisticRegression():
    """
        Class representing LogisticRegression
        Accepts LISTS of np.arrays ONLY!!!
    """

    def __init__(self, lam_1=0., lam_2=0, sample_good_ones=True):
        '''
        :lam_1 = L1 regularization parameter
        :lam_2 = L2 regularization parameter
        '''
        # hypothesis function
        self.expapprox = 30
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.sample_good_ones = sample_good_ones

        self.w = None

        # performance analysis
        self.fevals = 0
        self.gevals = 0
        self.adp = 0

    # sigmoid function
    def sigmoid(self, z):
        '''
        sigmoid function

        INPUTS:
        - self
        - z

        OUTPUS:
        - sigmoid z
        '''

        if math.isnan(z):
            z = 0
        else:
            z = np.sign(z) * min([np.abs(z), self.expapprox])

        return 1/(1.0+np.exp(-z))

    def h(self, w, X):
        '''
        h

        INPUTS:
        - self
        - w
        - X

        OUTPUTS:
        -
        '''
    #   print (type(X))
    #   print (type(X[0]))
    #   print (w)
    #   print (type(w))
    #   print ("debug:", np.multiply(w,X))
    
        return self.sigmoid(np.multiply(w, X).sum())

    def f(self, w, X, y):
        '''
        Loss functions as column vector

        INPUTS:
        - self
        - w : weights ?
        - X : data
        - y : targets

        OUTPUT:
        - loss function values
        '''

        hyp = self.h(w, X)
        self.fevals += 1

        return -y * np.log(hyp) - (1-y) * (np.log(1-hyp))+self.L_2(w)

    def L_2(self, w):
        '''
        L_2

        INPUTS:
        - self
        - w

        OUTPUT:
        -
        '''
        return 0.5 * self.lam_2 * (np.linalg.norm(w[1:])**2)

    # TODO:
    # Correct?
    def L_1(self, w):
        '''
        L_1

        INPUTS:
        - self
        - w

        OUTPUT:
        -
        '''

        return self.lam_1 * sum(map(abs, w[1:]))

    def F(self, w, X, y, lam=0.0):
        '''
        Overall objective function

        INPUTS:
        - self
        - w
        - X
        - y
        - lam

        OUTPUT:
        - Overall objective function
        '''

        #return sum(map(lambda t: self.f(w, t[0], t[1]), zip(X, y)))/len(X) +  self.L_2(w) 
        return sum([self.f(w,X[i],y[i]) for i in range(len(y))])/float(len(y))

    def g(self, w, X, y):
        '''
        Gradient of F

        INPUTS:
        - w
        - X
        - y

        OUTPUT:
        - gradient of f evaluated at exactly one sample
        '''

        # TODO: L_1 term correct ???
        # TODO
        hyp = self.h(w, X)
        self.gevals += 1
        #print "hyp-y, "(hyp-y).shape
        #print X
        return np.dot((hyp - y), X) + self.lam_2 * w  + self.lam_1 * w

    def G(self, w, X, Y):
        """
        :returns: stochastic gradient of the logistic regression problem
        """
        return sum([self.g(w, X[i], Y[i]) for i in range(len(X))])/len(X)

    def train(self, X, y, method='SQN'):
        '''
        Determine the regression variable w
        '''
        err_mes1 = "ERROR: Need at least one sample!"
        assert len(X) > 0, err_mes1

        err_mes2 = "ERROR: Sample and label list need to have same length!"
        assert len(X) == len(y), err_mes2

        if method == 'SQN':
            M = 10
            L = 10
            beta = 1.0
            batch_size = 10
            batch_size_H = 10
            max_iter = 1600
            sqn = SQN()
            # sqn.debug = True
            sqn.set_options({'dim': len(X[0]),
                             'max_iter': 45,
                             'batch_size': 10,
                             'beta': 10.,
                             'M': 10,
                             'batch_size_H': 10,
                             'L': 10,
                             'sampleFunction': self.sample_batch})
            self.w = sqn.solve(self.F, self.g, X=X, z=y)

            # self.w = SQN.solveSQN(self.F,
            #                       self.g,
            #                       X=X,
            #                       z=y,
            #                       w1=None,
            #                       dim=len(X[0]),
            #                       M=M,
            #                       L=L,
            #                       beta=beta,
            #                       batch_size=batch_size,
            #                       batch_size_H=batch_size_H,
            #                       max_iter=max_iter,
            #                       sampleFunction=self.sample_batch)

        else:
            raise NotImplementedError("ERROR: Method %s not implemented!" %method)

    def predict(self, X):
        """
        calculate the classification probability of samples

        INPUTS:
        - self
        - X : a list of samples

        OUTPUT
        - function
        """
        if len(np.shape(X)) < 2:
            X = [X]
        return map(lambda x: self.h(self.w, x), X)

    def get_sample(self, sampleList, X, z=None):
        '''
        get_sample

        INPUT:
        - self
        - sampleList
        - X
        - z

        OUTPUT:
        - return list of sample
        '''

        z_S = None if z is None else [z[i] for i in sampleList]

        return [X[i] for i in sampleList], z_S

    def sample_batch(self, w, X, z=None, b=None, r=None, debug=False):
        """
        returns a subsample X_S, y_S of the data, choosing only datapoints
        that are currently misclassified

        This function can be used in place of the 'generic' 
        sample_batch function in stochastic_tools

        Parameters:
            w: Regression variable
            X: training data
            z: Label
            b: parameter for desired max. subsample size (e.g. b=10)
            r: desired relative max. subsample size (e.g. r=.1)
        """
        if debug:
            print ("debug: ", b)

        err_mes1 = "Choose either absolute or relative sample size!"
        assert b is not None or r is not None, err_mes1

        err_mes2 = "Choose only one: Absolute or relative sample size!"
        assert (b is not None) != (r is not None), err_mes2

        # determine factual batch size
        if type(X) == type(list()):
            N = len(X)
        elif type(X) == type(int()) or type(X) == type(0.0):
            N = int(X)
        else:
            raise Exception("X is in the wrong format!" + str(type(X)))
        if b is not None:
            nSamples = b
        else:
            nSamples = r*N
        if nSamples > N:
            if debug:
                print ("Batch size larger than N, using whole dataset")
            nSamples = N

        # Find samples that are not classified correctly

        sampleList = []
        counter = 0
        while len(sampleList) < nSamples and counter < 10*b:
            random_index = np.random.randint(N)
            X_S, z_S = self.get_sample([random_index], X, z)
            if self.sample_good_ones or self.f(w, X_S[0], z_S[0]) > .1:
                sampleList.append(random_index)
            counter += 1

        # if not enough samples are found, we simply return a smaller sample!
        nSamples = len(sampleList)
        X_S, z_S = self.get_sample(sampleList, X, z)

        # Count accessed data points
        self.adp += nSamples

        if debug:
            print(X_S, z_S)

        return X_S, z_S
