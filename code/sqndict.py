from DictLearning.dictionary_learning import StochasticDictionaryLearning
from SQN.SQN import SQN

class SqnDictionaryLearning(StochasticDictionaryLearning):
           
        alpha = []
        
        def matrix_to_vector(self, D):
                v = []
                for i in range(D.range[0]):
                        for j in range(D.range[1]):
                                v.append(D[i,j])
                return np.array(v)
            
        def vector_to_matrix(self, v, nrow):
                D = np.matrix(0, nrow)
                i = 0
                j = 0
                for index in range(len(v)):
                    D[i,j] = v[index]
                    j += 1 
                    if j % nrow == 0:
                            i += 1
                            j = 0
                return D

        def f(self, d, X):
                self.D = self.vector_to_matrix(d)
                lam = 0
                self.alpha = [ subproblem(x) for x in X ]
                one_f = [ 0.5*np.linalg.norm(x - self.lasso_subproblem(x))**2 + lam*np.linalg.norm(a, ord=1) for x, a in zip(X, self.alpha) ]
                return sum(one_f) / len(one_f)
            
        def g(self, d, X, alpha):
                self.D = self.vector_to_matrix(d)
                for i in range(D.shape[0]):
                        for j in range(D.shape[1]):
                                one_g = [ x[j] * sum([D[i,k]*x[k] for k in range(D.shape[1])]) - a[i]*x[j]  for x, a in zip(X, self.alpha) ]
                return self.vector_to_matrix( sum(one_g)/len(one_g) )
                    
        
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
        if self.verbose > 10:
            print "Dimensions infos :"
            print "    n_samples :", n_s
            print "    D shape :", D.shape
            print "    A shape :", A.shape
            print "    B shape :", B.shape

        if self.verbose > 0:
            print "Running online dictionary learning"
            print str(self.n_iter), "iter to perform"
        
        sqn = SQN()
        sqn.solve(self.f, self.g, X)
        
        # Measure of running time
        dt_fit = time() - t_fit
        m_tsteps = np.mean(l_tsteps)

        if self.verbose > 0:
            print('fitting done in %.2fs.' % dt_fit)
            print('updating dic in %.2fs.' % m_tsteps)

        self.components = D


if __name__ == '__main__':
        
        dictlearner