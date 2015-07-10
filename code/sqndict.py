import itertools
import numpy as np
# from time import time
from DictLearning.dictionary_learning import StochasticDictionaryLearning
from SQN.SGD import SQN
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
from ProxEegTest import prox_meth_lr


class DictSQN(SQN):
        normalization = None

        def _perform_update(self, f_S, g_S, k=None):
                search_direction = self._get_search_direction(g_S)
                alpha = 0.001
                alpha = self._armijo_rule(f_S, g_S,
                                          search_direction,
                                          start=self.options['beta'],
                                          beta=.5,
                                          gamma=1e-2)

                self.w = self.w + np.multiply(alpha, search_direction)
                self.w = self.normalization(self.w)
                return self.w


class SqnDictionaryLearning(StochasticDictionaryLearning):

        # store lasso coefficients
        recon = []

        # store dictionaries updates
        updates = []

        def matrix_to_vector(self, D):
                return D.flat[:]
                """
                v = []
                for i in range(D.shape[0]):
                        for j in range(D.shape[1]):
                                v.append(D[i, j])
                return np.array(v)
                """

        def vector_to_matrix(self, v):
                D = v.reshape((len(v.flat)/self.n_components, self.n_components))
     #           for i in range(len(D)):
        #                D[i] = np.multiply(min(1.0, 1.0/np.linalg.norm(D[i])), D[i])
                return D
                """
                nrow = self.n_components
                D = np.zeros((len(v)/self.n_components, self.n_components))

                i = 0
                j = 0
                for index in range(len(v)):
                    D[i, j] = v[index]
                    j += 1
                    if j % nrow == 0:
                            i += 1
                            j = 0
                return D
                """

        def f(self, d, X, z=None):
                D = self.vector_to_matrix(d)
                # print D
                one_f = []
                # print np.array(self.recon.T[0].flat)
                # print "dot", D.dot(np.array(self.recon.T[0].flat))
                for x, a in zip(X, z.T):
                        x = np.asarray(x.flat)
                        a = np.asarray(a.flat)
                        l2_norm = 0.5 * np.linalg.norm(x - D.dot(a))**2
                        l1_norm = self.alpha * np.linalg.norm(a, ord=1)
                        one_f.append(l2_norm + l1_norm)
                return sum(one_f) / len(one_f)

        def finite_differences(self, d, X, z=None):
                print "differencing..."
                grad = np.zeros(d.shape)
                r = 0.001
                for i in range(len(d)):
                        step = d.copy()
                        step[i] += r
                        # print step-d
                        grad[i] = (self.f(step, X, z) - self.f(d, X, z))/r
                return grad

        def g_fast(self, d, X, z=None):
                D = self.vector_to_matrix(d)
                GD = np.multiply(0.0, D.copy())

                for x, a in zip(X, z.T):
                        a = np.array(a.flat)
                        x = np.array(x.flat)
                        GD_S = np.multiply(0.0, D.copy())

                        for j in range(D.shape[1]):
                            GD_S[:, j] = np.multiply(2.0 * a[j], D.dot(a)-x)

                        GD = GD + GD_S
                GD = np.multiply(1.0/len(X), GD)
                return self.matrix_to_vector(GD)

        def g(self, d, X, z=None):
                """
                print "g"
                print np.max(self.g_fast(d, X, z) - \
                 self.finite_differences(d, X, z))

                print np.max(self.g_comp_wise(d, X, z) - \
                 self.g_fast(d, X, z))
                """
                # return self.finite_differences(d, X, z)
                return self.g_fast(d, X, z)

        def normalization(self, d):
                D = self.vector_to_matrix(d)
                for col in range(D.shape[1]):
                        D[:, col] = np.multiply(min(1.0, 1.0/np.linalg.norm(D[:, col])), D[:, col])
                return(self.matrix_to_vector(D))

        def lasso_subproblem(self, Xt, comp):
                print "inside lasso"
                # 4: Sparse coding with LARS
                lars = LassoLars(alpha=self.alpha, verbose=False)

                lars.fit(comp, Xt)
                coef = lars.coef_
                # print coef
                coef = (np.asmatrix(coef)).T

                # Dimension control
                if self.verbose > 20:
                    print "coef shape :", coef.shape

                return coef

        def fit(self, X):
            '''
            This method runs online dictionary learning on data X and update
            self.components

            INPUTS :
            - self : sdl object
            - X : data
            '''
            if self.verbose > 0:
                print ""
                print "Running online dictionary learning"
                print str(self.n_iter), "iter to perform"

            print ""
            print "Intialize sqn"
            options = {'dim': len(X[0])*self.n_components,
                       'max_iter': self.n_iter,
                       'batch_size': self.batch_size,
                       'beta': 1.,
                       'M': 3,
                       'batch_size_H': 2000,
                       'L': 6,
                       'two_loop': True,
                       'updates_per_batch': self.max_iter}

            sqn = DictSQN(options)
            sqn.normalization = self.normalization
            # sqn.debug = True

            # initial dictionary : take some elements of X
            jd = np.random.randint(0, len(X[:]), self.n_components)
            self.components = X[jd, :].T
            self.updates.append(self.components)

            # initial position
            d = np.array(self.components.flat)
            sqn.set_start(w1=d)

            # check first dictionary
            if self.verbose > 20:
                print ""
                print "check first dictionary"
                print "self.components dim", self.components.shape

                print ""
                print "check first position"
                print "position dim", d.shape

            print ""
            print "max_iter to perform :", self.n_iter

            def draw_sample(X, z, w, b):

                    index_list = np.random.randint(0, len(X),
                                                   options['batch_size'])
                    Xt = np.asmatrix(X[index_list, :])
                    z = self.lasso_subproblem(Xt.T, self.vector_to_matrix(w))

                    return Xt, z
            sqn._draw_sample = draw_sample

            # loop to learn dictionary
            for k in itertools.count():

                print "\niter :", k + 1

                # dictionary reshaping
                self.components = self.vector_to_matrix(d)

                # update dictionary with sqn
                d = sqn.solve_one_step(self.f, self.g, X, None, k)
                print sqn.f_vals[-1]

                # transform dictionary into a matrix
                self.components = self.vector_to_matrix(d)

                # store each dictionary update
                self.updates.append(self.components)

                # condition to stop the loop
                if k > sqn.options['max_iter'] or sqn.termination_counter > 2:
                    iterations = k
                    break

            if iterations < sqn.options['max_iter']:
                print("Terminated successfully!")

            # plot successive dictionaries
            plt.show()

            print "SQN done!"


if __name__ == '__main__':
        print "main"
        sd = SqnDictionaryLearning()
        sd.n_components = 3
        D = np.matrix([[1, 2, 3], [4, 5, 6]])
        print D
        print sd.matrix_to_vector(D)
        print D.flat[:]
        v = sd.matrix_to_vector(D)
        print sd.vector_to_matrix(v)
        print v.reshape(2, len(v.flat)/2)
