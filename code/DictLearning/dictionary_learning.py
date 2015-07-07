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
    - test it on scikit learn ex.
    - write test dictionary learning script
    - add a verbose function
    - selection of best regularization parameter or inscrease batch size
    - should I add a runtime attribute ? => maybe not yet, maybe not here

    TODO :
    - select option : SQN or normal where would be the access point ?
"""


import numpy as np
import math
from time import time


class StochasticDictionaryLearning:
    '''
    This class implements dictionary learning

    Attributes:
    - n_components
    - option, select SQN method or normal method
    - alpha, regularization parameter for Lasso subproblem
    - n_iter, int, number of iterations
    - max_iter, int, number of iterations for dictionary update
    - batch_size, int, mini batch size
    - verbose, int, control verbosity of algorithm
    - components, dictionary atoms learnt


    Methods:
    __init__ :
    fit :
    set_params :
    get_params :
    print_attributes ?

    '''

    def __init__(self, n_components=100, option=None, alpha=0.001,
                 n_iter=10, max_iter=100, epsilon=0.0001, batch_size=3,
                 verbose=0):

        self.n_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.components = []
        self.epsilon = epsilon

        self.option = option
        self.verbose = verbose

    def print_attributes(self):
        print "n_components", self.n_components
        print "alpha", self.alpha
        print "n_iter", self.n_iter
        print "max_iter", self.max_iter
        print "batch_size", self.batch_size
        print "epsilon", self.epsilon

    def fit(self, X):
        '''
        This method runs online dictionary learning on data X and update
        self.components

        INPUTS :
        - self : sdl object
        - X : data
        '''

        # Measure of running time
        t_fit = time()
        l_tsteps = []

        # number of samples in x
        n_s = len(X[:, 0])

        # initial dictionary : take some elements of X
        jd = np.random.randint(0, n_s, self.n_components)

        # assign D to self.components
        self.components = X[jd, :].T

        # Dimensions of dicitonary
        [m, k] = self.components.shape

        # 1: initialization
        # A = np.zeros((k, k))
        # B = np.zeros((m, k))

        A = np.identity(k)
        B = X[jd, :].T

        '''
        Consider rescaling ?
        '''

        # dimension control
        if self.verbose > 10:
            print "Dimensions infos :"
            print "    n_samples :", n_s
            print "    D shape :", self.components.shape
            print "    A shape :", A.shape
            print "    B shape :", B.shape

        if self.verbose > 0:
            print "Running online dictionary learning"
            print str(self.n_iter), "iter to perform"

        t = 0
        run_control = True
        while t < self.n_iter and run_control is True:
            t += 1

            # verbosity control
            if self.verbose > 10:
                print "iter :", t

            # Measure running time of each iteration
            t_step = time()

            # Draw mini-batch
            j = np.random.randint(0, n_s, self.batch_size)
            Xt = X[j, :]
            Xt = np.asmatrix(Xt).T

            # dimension control
            if self.verbose > 20 and t == 1:
                print "    Xt shape :", Xt.shape

            # online dictionary learning algorithm called
            self.online_dictionary_learning(Xt, A, B, t)

            # Measure running time of each iteration
            dt_step = time() - t_step
            l_tsteps.append(dt_step)

            '''
            Q? : should I update D or self.components ?
            Q? : should I input X or mini_batch ? => mini-batch
            '''

        # Measure of running time
        dt_fit = time() - t_fit
        m_tsteps = np.mean(l_tsteps)

        if self.verbose > 0:
            print('fitting done in %.2fs.' % dt_fit)
            print('updating dic in %.2fs.' % m_tsteps)

        # self.components = D

    def online_dictionary_learning(self, Xt, A, B, t):
        '''
        This function perform online dictionary algorithm with data X and
        update self.components

        INPUTS:
        - self
        - Xt, data array
        - A, matrix
        - B, matrix
        - t, iter number

        '''

        # 4: Sparse coding with LARS
        coef = self.lasso_subproblem(Xt)

        # 5/6: Update A and B
        A, B = self.update_matrices(Xt, coef, A, B, t)
        if self.verbose > 20:
            print "det A:", np.linalg.det(A)

        self.dictionary_update(A, B)

    def lasso_subproblem(self, Xt):
        '''
        function which performs:
        - 4: Sparse coding with LARS

        INPUTS:
        - self
        - Xt, data array
        - A, matrix
        - B, matrix
        - t, iter number

        OUTPUT:
        - coef
        '''
        print "inside lasso"
        # 4: Sparse coding with LARS
        from sklearn.linear_model import LassoLars
        lars = LassoLars(alpha=self.alpha, verbose=False)

        # self.components = np.matrix([[8,2,3,4],[1,6,1,99]])
        # Xt = np.matrix([[3,1],[6,7]])
        # Xt[1,1] = 9999
        lars.fit(self.components, Xt)
        coef = lars.coef_
        # print coef
        coef = (np.asmatrix(coef)).T

        # Dimension control
        if self.verbose > 20:
            print "coef shape :", coef.shape

        return coef

    def update_matrices(self, Xt, coef, A, B, t):
        '''
        function which performs:
        - computing coefficient beta for step 5/6
        - 5: Update A
        - 6: Update B

        INPUTS:
        - self
        - Xt, data array
        - coef, solution of Lasso subproblem
        - A, matrix
        - B, matrix
        - t, iter number

        OUTPUTS:
        - A, matrix
        - B, matrix
        '''
        [m, k] = self.components.shape

        # computing coefficient beta for step 5/6
        if t < self.batch_size:
            theta = float(t * self.batch_size)
        else:
            theta = math.pow(self.batch_size, 2) + t - self.batch_size

        beta = (theta + 1 - self.batch_size) / (theta + 1)

        # 5: Update A
        a = np.zeros((k, k))
        for i in range(0, self.batch_size):
            a = a + (coef[:, i]).dot(coef[:, i].T)
        A = beta * A + a

        # 6: Update B
        b = np.zeros((m, k))
        for i in range(0, self.batch_size):
            b = b + Xt[:, i].dot(coef[:, i].T)
        B = beta * B + b

        return A, B

    def dictionary_update(self, A, B):
        '''
        Dictionary update : updates self.components

        INPUTS:
        - self
        - D, dictionary
        - A, matrix
        - B, matrix

        '''

        [m, k] = self.components.shape

        c = 0  # counter
        cv = False  # convergence or stop indicator

        # 2: loop to update each column
        while cv is not True:

            # keep a trace of previous dictionary
            D_old = np.zeros((m, k))
            for i in range(0, m):
                for j in range(0, k):
                    D_old[i, j] = self.components[i, j]

            for j in range(0, k):

                # 3: Update the j-th column of d
                s_A = np.sum(A[:, j])
                s_B = np.sum(B[:, j])
                a_jj = A[j, j]

                # print "s_A", s_A
                # print "s_B", s_B
                # print "a_jj", a_jj
                # print ""

                if s_A + s_B == 0 and a_jj == 0:
                    u = 1 + np.asmatrix(self.components[:, j]).T
                    if self.verbose > 20:
                        print "0 case"
                else:
                    u = (1 / A[j, j]) * (B[:, j] - self.components.dot(A[:, j]))
                    u = u + np.asmatrix(self.components[:, j]).T
                    if self.verbose > 20:
                        print "normal case"

                # renormalisation
                renorm = max(np.linalg.norm(u), 1)
                u = np.divide(u, renorm)

                '''
                What if u == 0 ?
                '''

                for p in range(0, m):
                    self.components[p, j] = u[p]

            # counter update
            c += 1

            # compute differences between two updates
            grad = self.components - D_old
            crit = np.linalg.norm(grad)

            # check convergence
            if crit < self.epsilon:
                cv = True
            if c > self.max_iter:
                cv = True

        if c == self.max_iter and self.verbose > 10:
            print "Dictionary Updating Algo reached max number of iterations"
            print "Consider higher max number of interations"

        # 6: Return updated dictionary


if __name__ == "__main__":
    sdl = StochasticDictionaryLearning()
    sdl.print_attributes()
