import itertools
import numpy as np
from time import time
from DictLearning.dictionary_learning import StochasticDictionaryLearning
from SQN.SQN import SQN
from scipy.optimize import minimize

class SqnDictionaryLearning(StochasticDictionaryLearning):

        recon = []

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
                return v.reshape((len(v)/self.n_components, self.n_components))
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
                D = d.reshape(len(d)/self.n_components, self.n_components)
               # print D
                one_f = []
                #print np.array(self.recon.T[0].flat)
                #print "dot", D.dot(np.array(self.recon.T[0].flat))
                for x, a in zip(X, self.recon.T):
                        x = np.asarray(x.flat)
                        a = np.asarray(a.flat)
                        l2_norm = 0.5 * np.linalg.norm(x - D.dot(a))**2
                        l1_norm = self.alpha * np.linalg.norm(a, ord=1) 
                        one_f.append( l2_norm + l1_norm )
                return sum(one_f) / len(one_f)

        def finite_differences(self, d, X, z=None):
                grad = np.zeros(d.shape)
                r = 0.001
                for i in range(len(d)):
                        step = d.copy()
                        step[i] += r
                        #print step-d
                        grad[i] = (self.f(step, X, z) - self.f(d, X, z))/r
                        #print grad[i]
                return grad
            
        def g_fast(self, d, X, z=None):
                D = d.reshape(len(d)/self.n_components, self.n_components)
                grad = np.zeros(d.shape)
                for x, a in zip(X, self.recon.T):
                    a = np.array(a.flat)
                    x = np.array(x.flat)
                    Dx = D.T.dot(x)
                    grad += (np.outer(Dx, x) - np.outer(a, x)).flat[:]
            
                return np.multiply(1.0/len(X), grad) 
            
        def g_comp_wise(self, d, X, z=None):
                D = d.reshape(len(d)/self.n_components, self.n_components)
                grad = np.zeros(d.shape)
                for x, a in zip(X, self.recon.T):
                    a = np.array(a.flat)
                    x = np.array(x.flat)
                    index = 0
                    for i in range(D.shape[1]):
                            S = 0.0
                            for k in range(D.shape[0]):
                                    S += D[k,i] * x[k]
                            for j in range(D.shape[0]):
                                    grad[index] += x[j] * S - a[i]*x[j]
                                    index += 1
                return np.multiply(1.0/len(X), grad)
                
        def g(self, d, X, z=None):
                """
                print "g"
                print np.max(self.g_fast(d, X, z) - self.finite_differences(d, X, z))
                print np.max(self.g_comp_wise(d, X, z) - self.finite_differences(d, X, z))
                print np.max(self.g_comp_wise(d, X, z) - self.g_fast(d, X, z))
                """
                #print self.finite_differences(d, X, z)
                #return self.finite_differences(d, X, z)
                return self.g_fast(d, X, z)
        
        
        """
        def lasso_subproblem(self, X):
                D = self.components
                zielfun = lambda a, x: 0.5*np.linalg.norm(x - D.dot(a))**2
                a = np.zeros( (D.shape[1], 1) )
                alpha = [ minimize(lambda w: zielfun(w, x), a)['x'] for x in X.T ]
                return np.asmatrix(alpha)
        """                             
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
                       'M': 5,
                       'batch_size_H': 10,
                       'L': 5,
                       'normalize': True,
                       'updates_per_batch': 30}
            
            sqn = SQN(options)

            # initial dictionary : take some elements of X
            jd = np.random.randint(0, len(X[:]), self.n_components)
            self.components = X[jd, :].T
            sqn._armijo_rule = lambda f,g,x,s, start, beta, gamma: 0.001
            sqn.set_start(w1 = np.array(self.components.flat))
            d = sqn.get_position()
            #print d
            for k in itertools.count():
                print k
                j = np.random.randint(0, len(X), self.n_components)#sqn.options['batch_size'])
                Xt = X[j, :]
                Xt = np.asmatrix(Xt)
                    
                def draw_sample(w, X, z, b):
                    return Xt, None
                sqn.draw_sample = draw_sample
                
                self.components = d.reshape(len(d)/self.n_components, self.n_components)
                #print Xt.T
                
                self.recon = self.lasso_subproblem(Xt.T)
                
                #print Xt
                print self.recon
                # TODO: WHY???
                break
                d = sqn.solve_one_step(self.f, self.g, X, None, k)
                #print d[1:10]
                #self.recon = []
                if k > sqn.options['max_iter'] or sqn.termination_counter > 4:
                    iterations = k
                    break

            if iterations < sqn.options['max_iter']:
                print("Terminated successfully!")
            
            print "SQN done!"

if __name__ == '__main__':
        print "main"
        sd = SqnDictionaryLearning()
        sd.n_components = 3
        D = np.matrix([[1,2,3],[4,5,6]])
        print D
        print sd.matrix_to_vector(D)
        print D.flat[:]
        v = sd.matrix_to_vector(D)
        print sd.vector_to_matrix(v)
        print v.reshape(2, len(v)/2)