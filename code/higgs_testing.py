
# from multiprocessing import Pool, Process
import itertools


"""
Logistic Regression
"""
from SQN.LogisticRegression import LogisticRegression
from SQN.LogisticRegressionTest import LogisticRegressionTest

from SQN.NewSQN import SQN
from SQN.SGD import SGD

# from SQN.PSQN import PSQN

import numpy as np
import timeit

import data.datasets as datasets

import sys

from SQN import stochastic_tools

"""
SQN
"""

def print_f_vals(sqn, options, folderpath, testcase=None, rowlim=None):
    
    
    print("\nSQN, Higgs-Dataset\n")

    logreg = LogisticRegression(lam_1=0.0, lam_2=0.0)
    logreg.get_sample = lambda l, X, z: datasets.get_higgs_mysql(l)
    sqn.set_start(dim=sqn.options['dim'])
    w = sqn.get_position()

    sqn.set_options({'sampleFunction': logreg.sample_batch})
    X, z = None, None
    
    if folderpath is not None:
        ffile = open(folderpath + "%d_%d.txt" %(sqn.options['batch_size'], sqn.options['batch_size_H']), "w+")
    
    sep = ","
    
    results = []
    
    f_evals = []
    for k in itertools.count():

        w = sqn.solve_one_step(logreg.F, logreg.G, k=k)
        #print k
        
        # X_S, z_S = sqn._draw_sample(sqn.options['N'], b = 100)
        f_evals.append(sqn.f_vals[-1])
        #print(logreg.F(w, X_S, z_S))
        
        
        if k%20 == 0 and sqn._is_stationary():
                print sqn._get_test_variance()
                print sqn.options['batch_size'], sqn.options['batch_size_H']
    #            sqn.set_options({'batch_size': sqn.options['batch_size']+2, 'batch_size_H': sqn.options['batch_size_H']+1})

        results.append([k, logreg.fevals, logreg.gevals, logreg.adp, sqn.f_vals[-1], sqn.g_norms[-1]])
        
        if folderpath is not None:
            line = sep.join([ str(r) for r in results[-1] ])[:-1] + "\n"
            ffile.write(line)
        else:
            print(k, logreg.adp, "%0.2f" % float(sqn.f_vals[-1]))
        if k > sqn.options['max_iter'] or sqn.termination_counter > 4:
            iterations = k
            break
    if folderpath is not None:
        ffile.close()
    
    return results
    

"""
Main
"""
if __name__ == "__main__":
    
        """
        Runs SQN-LogReg on the Higgs-Dataset,
        which is a 7.4GB csv file for binary classification
        that can be obtained here:
        https://archive.ics.uci.edu/ml/datasets/HIGGS
        the file should be in <Git Project root directory>/datasets/
        """
        rowlim = 5e6
        batch_size = 100
        options = {'dim':29, 'N':rowlim, 'L': 4, 'max_iter': 1000, 'batch_size': batch_size, 'batch_size_H': 4000, 'beta':2, 'M':4, 'updates_per_batch': 2, 'testinterval':20}
        
        folderpath = "../outputs/"
        folderpath = None
        
        batch_sizes = [100, 500, 1000, 10000]        
        batch_sizes = [500]
        result_list = []
        n_Samples = 1
        batch_size = 10000
        for i in range(n_Samples):
                options['batch_size'] = batch_size
                sqn = SGD(options)
                result_list.append(print_f_vals(sqn, options, folderpath))
        results = np.matrix(result_list[0])
        for i in range(1, len(result_list)):
                results += np.matrix(result_list[i])
        results = np.multiply(1.0/len(result_list), results)
       # for r in results:
         #       print list(r.flat)
        testcase = "sql"
        """
        for batch_size in batch_sizes:
            options['batch_size'] = batch_size
            # Select method
            sqn = SQN(options)
            #sqn = PSQN(options)
            f = lambda b: print_f_vals(sqn, options, folderpath)
            print(f(batch_size))
#            p = Process(target=f, args=(batch_size,))
#            p.start()
        """