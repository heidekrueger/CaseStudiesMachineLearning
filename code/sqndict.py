import numpy as np
from time import time
from DictLearning.dictionary_learning import StochasticDictionaryLearning
from SQN.SQN import SQN


class SqnDictionaryLearning(StochasticDictionaryLearning):

        recon = []

        def matrix_to_vector(self, D):

                v = []
                for i in range(D.range[0]):
                        for j in range(D.range[1]):
                                v.append(D[i, j])
                return np.array(v)

        def vector_to_matrix(self, v):
                print "v:", len(v)

                nrow = self.n_components
                D = np.zeros((self.n_components, len(v)))

                print D.shape

                i = 0
                j = 0
                for index in range(len(v)):
                    D[i, j] = v[index]
                    j += 1
                    if j % nrow == 0:
                            i += 1
                            j = 0
                return D

        def f(self, d, X, z=None):
            print "shape X:", len(X)

            self.recon = []

            if not isinstance(self.recon, list):
                        self.recon = [self.recon]

            if not isinstance(X, list):
                        X = [X]

            # self.components = self.vector_to_matrix(d)

            print "doing lasso"
            self.recon = [self.lasso_subproblem(np.asmatrix(x).T) for x in X]
            print "lasso done"

            one_f = [0.5 * np.linalg.norm(x - (self.components).dot(a))**2 + self.alpha * np.linalg.norm(a, ord=1) for x, a in zip(X, self.recon) ]
            return sum(one_f) / len(one_f)

        def g(self, d, X, z=None):

                if not isinstance(self.recon, list):
                        self.recon = [self.recon]
                if not isinstance(X, list):
                        X = [X]

                # self.D = self.vector_to_matrix(d)
                # self.recon = [self.lasso_subproblem(x) for x in X]
                # x = X[0]
                # a = self.recon[0]
                # one_g = []
                # print a
                # for i in range(self.D.shape[0]):
                #         for j in range(self.D.shape[1]):
                #                 S = 0.0
                #                 for k in range(self.D.shape[1]):
                #                         S += self.D[i, k] * x[k]
                #                 one_g.append(x[j] * S - a[i]*x[j])
                # return self.vector_to_matrix(sum(one_g)/len(one_g))
                return 0

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
            self.components = X[jd, :].T


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
            if self.verbose > 10:
                print "Dimensions infos :"
                print "    n_samples :", n_s
                print "    D shape :", D.shape
                print "    A shape :", A.shape
                print "    B shape :", B.shape

            if self.verbose > 0:
                print "Running online dictionary learning"
                print str(self.n_iter), "iter to perform"

            def sample_batch(w, X, z, b):
                    j = np.random.randint(0, n_s, b)
                    Xt = X[j, :]
                    Xt = np.asmatrix(Xt).T
                    X_S = []
                    for i in range(X.shape[0]):
                            X_S.append(np.array(X[i, :].flat))
                    return X_S, None

            options = {'dim': len(X[0, :]),
                       'max_iter': self.max_iter,
                       'batch_size': self.batch_size,
                       'beta': 10.,
                       'M': 10,
                       'batch_size_H': 10,
                       'L': 10,
                       'sampleFunction': sample_batch}

            sqn = SQN(options)
            # sqn.set_options(self.options)
            sqn.solve(self.f, self.g, X)

            # Measure of running time
            dt_fit = time() - t_fit
            m_tsteps = np.mean(l_tsteps)

            if self.verbose > 0:
                print('fitting done in %.2fs.' % dt_fit)
                print('updating dic in %.2fs.' % m_tsteps)

            self.components = D


if __name__ == '__main__':
        print "main"
