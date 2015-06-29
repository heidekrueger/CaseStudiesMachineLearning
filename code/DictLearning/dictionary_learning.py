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


    TODO :
    - add a verbose function
    - selection of best regularization parameter or inscrease batch size
    - select option : SQN or normal where would be the access point ?
    - should I add a runtime attribute ? => maybe not yet, maybe not here
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
                 n_iter=10, max_iter=100, epsilon=0.0001, batch_size=200,
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
        D = X[jd, :].T
        # D = np.random.rand(len(X[0, :]), self.n_components)
        '''
        Q? : should I work with self.components ??
        '''

        # Dimensions of dicitonary
        [m, k] = D.shape

        # 1: initialization
        # A = np.zeros((k, k))
        # B = np.zeros((m, k))

        A = np.identity(k)
        B = X[jd, :].T

        '''
        Consider rescaling ?
        '''

        # dimension control
        if self.verbose > 0:
            print "Dimensions infos :"
            print "    n_samples :", n_s
            print "    D shape :", D.shape
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
            D = self.online_dictionary_learning(Xt, D, A, B, t)

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

        self.components = D

    def online_dictionary_learning(self, Xt, D, A, B, t):
        '''
        This function perform online dictionary algorithm with data X and
        update D or self.components

        INPUTS:
        - self
        - Xt, data array
        - D, dictionary
        - A, matrix
        - B, matrix
        - t, iter number

        Q? : should I update D or self.components ??
        '''

        # 4: Sparse coding with LARS
        coef = self.lasso_subproblem(Xt, D, A, B, t)
        print "coef :", np.sum(coef)

        # 5/6: Update A and B
        A, B = self.update_matrices(Xt, D, coef, A, B, t)
        z_control = np.linalg.det(A)
        print z_control

        D = self.dictionary_update(D, A, B)

        # return Dictionary
        return D

    def lasso_subproblem(self, Xt, D, A, B, t):
        '''
        function which performs:
        - 4: Sparse coding with LARS

        INPUTS:
        - self
        - Xt, data array
        - D, dictionary
        - A, matrix
        - B, matrix
        - t, iter number

        OUTPUT:
        - coef
        '''

        # 4: Sparse coding with LARS
        from sklearn.linear_model import LassoLars
        lars = LassoLars(alpha=self.alpha)

        lars.fit(D, Xt)
        coef = lars.coef_
        coef = (np.asmatrix(coef)).T

        # Dimension control
        if self.verbose > 20:
            print "coef shape :", coef.shape

        return coef

    def update_matrices(self, Xt, D, coef, A, B, t):
        '''
        function which performs:
        - computing coefficient beta for step 5/6
        - 5: Update A
        - 6: Update B

        INPUTS:
        - self
        - Xt, data array
        - D, dictionary
        - coef, solution of Lasso subproblem
        - A, matrix
        - B, matrix
        - t, iter number

        OUTPUTS:
        - A, matrix
        - B, matrix
        '''
        [m, k] = D.shape

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

    def dictionary_update(self, D, A, B):
        '''
        Dictionary update

        INPUTS:
        - self
        - D, dictionary
        - A, matrix
        - B, matrix


        OUTPUT:
        - D, updated dictionary
        '''

        [m, k] = D.shape

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
                s_A = np.sum(A[:, j])
                s_B = np.sum(B[:, j])
                a_jj = A[j, j]

                # print "s_A", s_A
                # print "s_B", s_B
                # print "a_jj", a_jj
                # print ""

                if s_A + s_B == 0 and a_jj == 0:
                    u = 1 + np.asmatrix(D[:, j]).T
                    if self.verbose > 20:
                        print "0 case"
                else:
                    u = (1 / A[j, j]) * (B[:, j] - D.dot(A[:, j]))
                    u = u + np.asmatrix(D[:, j]).T
                    if self.verbose > 20:
                        print "normal case"

                # renormalisation
                renorm = max(np.linalg.norm(u), 1)
                u = np.divide(u, renorm)

                '''
                What if u == 0 ?
                '''

                for p in range(0, m):
                    D[p, j] = u[p]

            # counter update
            c += 1

            # compute differences between two updates
            grad = D - D_old
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
        return D


# def algorithm1(x, n_components=100, alpha=0.01, n_iter=30, batch_size=3,
#                verbose=0):
#     '''
#     Online dictionary learning algorithm

#     INPUTS:
#     - x : (n_samples, m) array like, data
#     - l, regularization parameter
#     - n_components, number of components of dictionary
#     - n_iter, int, number of iterations
#     - batch_size, int, mini batch size
#     - verbose, int, control verbosity of algorithm

#     OUTPUTS:
#     - D : (m, k) array like, learned dictionary
#     '''

#     n_s = len(x[:, 0])  # number of samples in x

#     D = np.random.rand(len(x[0, :]), n_components)  # initial dictionary
#     print D.shape

#     # Dimensions of dicitonary
#     [m, k] = D.shape

#     # 1: initialization
#     A = np.zeros((k, k))
#     B = np.zeros((m, k))

#     '''
#     First iteration in a while loop so that matrix A in not empty
#     - if empty => change l
#     - if empty => change batch size

#     So that we ensure that no problem occurs in algo2
#     '''

#     '''
#     Loop
#     for t in range(2, n_iter + 1):

#     '''

#     # 2: Loop
#     for t in range(1, n_iter + 1):

#         # 3: Draw xj from x
#         j = np.random.randint(0, n_s, batch_size)
#         xt = x[j, :]
#         xt = np.asmatrix(xt).T

#         # 4: Sparse coding with LARS
#         from sklearn.linear_model import LassoLars
#         lars = LassoLars(alpha=alpha)
#         # Lars LassoLars

#         lars.fit(D, xt)
#         coeff = lars.coef_
#         coeff = (np.asmatrix(coeff)).T

#         # computing coefficient beta for step 5/6
#         if t < batch_size:
#             theta = float(t * batch_size)
#         else:
#             theta = math.pow(batch_size, 2) + t - batch_size

#         beta = (theta + 1 - batch_size) / (theta + 1)

#         # 5: Update A
#         a = np.zeros((k, k))
#         for i in range(0, batch_size):
#             a = a + (coeff[:, i]).dot(coeff[:, i].T)
#         A = beta * A + a

#         # 6: Update B
#         b = np.zeros((m, k))
#         for i in range(0, batch_size):
#             b = b + xt[:, i].dot(coeff[:, i].T)
#         B = beta * B + b

#         # Compute new dictionary update
#         # D = algorithm2(D, A, B)

#     # 9 : Return learned dictionary
#     return D


# def algorithm2(D, A, B, c_max=3, eps=0.00001):
#     '''
#     Dictionary update

#     INPUTS:
#     - D, (m, k), input dictionary
#     - A, (k, k)
#     - B, (m, k)
#     - c_max, int, max number of iterations
#     - eps, float, stopping criterion

#     OUTPUT:
#     - D, updated dictionary
#     '''

#     m = len(D[:, 0])
#     k = len(D[0, :])

#     c = 0  # counter
#     cv = False  # convergence or stop indicator

#     # 2: loop to update each column
#     while cv is not True:

#         # keep a trace of previous dictionary
#         D_old = np.zeros((m, k))
#         for i in range(0, m):
#             for j in range(0, k):
#                 D_old[i, j] = D[i, j]

#         for j in range(0, k):

#             # 3: Update the j-th column of d
#             u = (1 / A[j, j]) * (B[:, j] - D.dot(A[:, j]))
#             u = u + np.asmatrix(D[:, j]).T

#             # renormalisation
#             renorm = max(np.linalg.norm(u), 1)
#             u = np.divide(u, renorm)

#             for p in range(0, m):
#                 D[p, j] = u[p]

#         # counter update
#         c = c + 1

#         # compute differences between two updates
#         grad = D - D_old
#         crit = np.linalg.norm(grad)

#         # check convergence
#         if crit < eps:
#             cv = True
#         if c > c_max:
#             cv = True

#     if c == c_max:
#         print "Dictionary Updating Algo reached max number of iterations"
#         print "Consider higher max number of interations"

#     # 6: Return updated dictionary
#     return D


if __name__ == "__main__":
    sdl = StochasticDictionaryLearning()
