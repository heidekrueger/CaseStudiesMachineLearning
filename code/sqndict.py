import itertools
import numpy as np
# from time import time
from DictLearning.dictionary_learning import StochasticDictionaryLearning
from SQN.SGD import SQN
import matplotlib.pyplot as plt


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
                for x, a in zip(X, self.recon.T):
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
                        # print grad[i]
                return grad

        def g_fast(self, d, X, z=None):
                D = self.vector_to_matrix(d)
                grad = np.zeros(d.shape)
                for x, a in zip(X, self.recon.T):
                    a = np.array(a.flat)
                    x = np.array(x.flat)
                    Dx = D.T.dot(x)
                    grad += (np.outer(Dx, x) - np.outer(a, x)).flat[:]

                return np.multiply(1.0/len(X), grad)

        def g_comp_wise(self, d, X, z=None):
                D = self.vector_to_matrix(d)
                grad = np.zeros(d.shape)
                for x, a in zip(X, self.recon.T):
                    a = np.array(a.flat)
                    x = np.array(x.flat)
                    index = 0
                    for i in range(D.shape[1]):
                            S = 0.0
                            for k in range(D.shape[0]):
                                    S += D[k, i] * x[k]
                            for j in range(D.shape[0]):
                                    grad[index] += x[j] * S - a[i]*x[j]
                                    index += 1
                return np.multiply(1.0/len(X), grad)

        def g(self, d, X, z=None):
                """
                print "g"
                print np.max(self.g_fast(d, X, z) - \
                 self.finite_differences(d, X, z))
                print np.max(self.g_comp_wise(d, X, z) - \
                 self.finite_differences(d, X, z))
                print np.max(self.g_comp_wise(d, X, z) - \
                 self.g_fast(d, X, z))
                """
                # print self.finite_differences(d, X, z)
                # return self.finite_differences(d, X, z)
                return self.finite_differences(d, X, z)

        """
        def lasso_subproblem(self, X):
                D = self.components
                zielfun = lambda a, x: 0.5*np.linalg.norm(x - D.dot(a))**2
                a = np.zeros( (D.shape[1], 1) )

                alpha = [ minimize(lambda w: zielfun(w, x), a) \
                ['x'] for x in X.T ]

                return np.asmatrix(alpha)
        """
        
        def normalization(self, d):
                print("YEI")
                D = self.vector_to_matrix(d)
                print D.shape
                for col in range(D.shape[1]):
                        D[:,col] = np.multiply(min(1.0, 1.0/np.linalg.norm(D[:,col])), D[:,col])
                return(self.matrix_to_vector(D))
                        
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

            def sample_batch(w, X, z, b):
                n_s = len(X[:, 0])
                j = np.random.randint(0, n_s, b)
                X_S = []
                for i in j:
                    X_S.append(X[i])
                return X_S, None

            def stochastic_gradient(g, w, X=None, z=None):
                    return g(w, X, z)

            options = {'dim': len(X[0])*self.n_components,
                       'max_iter': self.max_iter,
                       'batch_size': self.batch_size,
                       'beta': 1.,
                       'M': 5,
                       'batch_size_H': 20,
                       'L': 50000,
                       'normalize': True,
                       'normalization': self.normalization,
                       'updates_per_batch': 20}

            print ""
            print "Intialize sqn"
            sqn = SQN(options)

            # initial dictionary : take some elements of X
            jd = np.random.randint(0, len(X[:]), self.n_components)
            self.components = X[jd, :].T
            self.updates.append(self.components)

            # sqn._armijo_rule = lambda f, g, s, start, beta, gamma: 0.001
            sqn.set_start(w1=np.array(self.components.flat))

            # initial position
            d = sqn.get_position()

            # check first dictionary
            if self.verbose > 20:
                print ""
                print "check first dictionary"
                print "self.components dim", self.components.shape

                print ""
                print "check first position"
                print "position dim", d.shape

            print ""
            print "max_iter to perform :", self.max_iter

            # loop to learn dictionary
            for k in itertools.count():

                print ""
                print "iter :", k + 1

                # Create batch for the iteration
                j = np.random.randint(0, len(X), options['batch_size'])
                Xt = X[j, :]
                Xt = np.asmatrix(Xt)

                # Control batch dimension
                if self.verbose > 20:
                    print ""
                    print "check subsample dim"
                    print "subsample dim", Xt.shape

                # Draw sample function for sqn class
                """
                QUESTIONS :
                - what does it do ?
                - Should we remove it ?
                """
                def draw_sample(w, X, z, b):
                    return Xt, None

                sqn.draw_sample = draw_sample

                # dictionary reshaping
                self.components = self.vector_to_matrix(d)

                # check dictionary shape
                if self.verbose > 20:
                    print ""
                    print "check dictionary dim"
                    print "dictionary dim", self.components.shape

                # Solve first sub problem
                """
                QUESTIONS :
                - should we use prox methd from now on ?
                - Do we ask Fin to implement it ?
                """
                self.recon = self.lasso_subproblem(Xt.T)

                # update dictionary with sqn
                d = sqn.solve_one_step(self.f, self.g, X, None, k)

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

            self.components = self.vector_to_matrix(d)

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
