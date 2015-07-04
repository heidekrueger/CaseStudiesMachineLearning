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

        def f(self, d, X, z=None):
            #print "shape X:", len(X), len(d), d.shape

            self.recon = []

            if not isinstance(self.recon, list):
                        self.recon = [self.recon]

            if not isinstance(X, list):
                        X = [X]

            self.components = self.vector_to_matrix(d)

            #print "doing lasso"
            if len(self.recon) == 0:
                    self.recon = [self.lasso_subproblem(np.asmatrix(x).T) for x in X]
            #print "lasso done"

            one_f = [0.5 * np.linalg.norm(x - (self.components).dot(a))**2 + self.alpha * np.linalg.norm(a, ord=1) for x, a in zip(X, self.recon) ]
            return sum(one_f) / len(one_f)

        def g(self, d, X, z=None):

                if not isinstance(self.recon, list):
                        self.recon = [self.recon]
                if not isinstance(X, list):
                        X = [X]
                
                self.components = self.vector_to_matrix(d)

                if len(self.recon) == 0:
                        print "doing lasso"
                        self.recon = [self.lasso_subproblem(np.asmatrix(x).T) for x in X]

                #print "lasso done"
                x = X[0]
                a = np.array(self.recon[0].flat)
                one_g = []
                #print self.components.shape, x.shape
                grad = None
                for x, a in zip(X, self.recon):
                    a = np.array(a.flat)
                    for i in range(self.components.shape[1]):
                        for j in range(self.components.shape[0]):
                                S = 0.0
                                for k in range(self.components.shape[0]):
                        #                print i, k, self.components.shape
                                        S += self.components[k,i] * x[k]
                                #print S, x[j], a[i], x[j]
                                one_g.append(x[j] * S - a[i]*x[j])
                    
                    if grad is None:
                            grad = np.array(one_g)
                    else:
                            grad += np.array(one_g)
                return np.multiply(len(X), grad)
                

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
                    X_S = []
                    for i in j:
                            X_S.append(X[i])
                    return X_S, None
            def stochastic_gradient(g, w, X=None, z=None):
                    return g(w, X, z) 
            
            options = {'dim': len(X[0])*self.n_components,
                       'max_iter': self.max_iter,
                       'batch_size': self.batch_size,
                       'beta': 10.,
                       'M': 10,
                       'batch_size_H': 10,
                       'L': 10,
                       'sampleFunction': sample_batch}
            
            sqn = SQN(options)
         #   sqn.stochastic_gradient = stochastic_gradient

            # sqn.set_options(self.options)
            #sqn.solve(self.f, self.g, X)
            "Please provide either a data set or a sampling function"

            sqn.set_start(dim=sqn.options['dim'])
            import itertools
            for k in itertools.count():
                print k
                w = sqn.solve_one_step(self.f, self.g, X, None, k)
                self.recon = []
                if k > sqn.options['max_iter'] or sqn.termination_counter > 4:
                    iterations = k
                    break

            if iterations < sqn.options['max_iter']:
                print("Terminated successfully!")


            
            print "SQN done!"

            self.components = D

if __name__ == '__main__':
        print "main"
