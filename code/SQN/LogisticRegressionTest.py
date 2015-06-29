import numpy as np

from LogisticRegression import LogisticRegression

class LogisticRegressionTest(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)

    def testF(self, X, y):
        logreg = LogisticRegression(lam_2 = 0.5)
        logreg.train(X, y)
        print("f complete")
        print(logreg.f(logreg.w, X[0], y[0]))
        print("f for first entry")
        print(logreg.f(logreg.w, X[0], y[0]))
        print("F")
        print(logreg.F(logreg.w,X,y))
        print("g ")
        print(logreg.g(logreg.w, X[0], y[0]))
    
    def test_classification(self, X, y):
        logreg = LogisticRegression(lam_2 = 0.5)
        logreg.train(X, y)
        print("predict", logreg.predict(X[0]) )
        print("error:", sum( (np.array([ logreg.predict(x) for x in X]) -np.array( y) )**2))
        print("F:", logreg.F(logreg.w, X, y))
        print("w:", logreg.w)
        
        print(logreg.fevals, logreg.gevals, logreg.adp)
        
if __name__ == '__main__':
    
    lrt = LogisticRegressionTest()
    lrt.testF( [ np.array([1,0]), np.array([0,1])], [0,1] )
    lrt.test_classification( [ np.array([1,0]), np.array([0,1])], [0,1] )