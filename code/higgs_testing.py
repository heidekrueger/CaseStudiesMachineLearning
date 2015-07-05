
# from multiprocessing import Pool, Process
import itertools


"""
Logistic Regression
"""
from SQN.LogisticRegression import LogisticRegression
from SQN.LogisticRegressionTest import LogisticRegressionTest

from SQN.SQN import SQN
# from SQN.PSQN import PSQN

import numpy as np
import timeit

import data.datasets as datasets

import sys

from SQN import stochastic_tools

"""
SQN
"""

def print_f_vals(testcase, rowlim, options, folderpath, sqn):

    print("\nSQN, Higgs-Dataset\n")

    logreg = LogisticRegression(lam_1=0.0, lam_2=0.0)
    logreg.get_sample = lambda l, X, z: datasets.get_higgs_mysql(l)
    sqn.set_start(dim=sqn.options['dim'])
    w = sqn.get_position()

    sqn.set_options({'sampleFunction': logreg.sample_batch})
    X, z = None, None
    
    if folderpath is not None:
        ffile = open(folderpath + "%d_%d.txt" %(sqn.options['batch_size'], sqn.options['batch_size_H']), "w+")
    
    f_evals = []
    for k in itertools.count():

        w = sqn.solve_one_step(logreg.F, logreg.G, k=k)
        print k
        
        # X_S, z_S = sqn._draw_sample(sqn.options['N'], b = 100)
        f_evals.append(sqn.f_vals[-1])
        #print(logreg.F(w, X_S, z_S))
        
        
        if k%20 == 0 and sqn._is_stationary():
                print sqn._get_test_variance()
                print sqn.options['batch_size'], sqn.options['batch_size_H']
    #            sqn.set_options({'batch_size': sqn.options['batch_size']+2, 'batch_size_H': sqn.options['batch_size_H']+1})

        sep = ","

        line = sep.join([str(logreg.fevals), str(logreg.gevals), str(logreg.adp), str(sqn.f_vals[-1]), str(sqn.g_norms[-1])])
        line = line[:-1] + "\n"

        if folderpath is not None:
            ffile.write(line)
        else:
            print(k, logreg.adp, "%0.2f" % float(sqn.f_vals[-1]))
        if k > sqn.options['max_iter'] or sqn.termination_counter > 4:
            iterations = k
            break
    if folderpath is not None:
        ffile.close()
    

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
        options = {'dim':29, 'N':rowlim, 'L': 4, 'max_iter': 5000, 'batch_size': batch_size, 'batch_size_H': 10, 'beta':10, 'M':10, 'updates_per_batch': 3, 'testinterval':30}
        
        folderpath = "../outputs/"
        #folderpath = None
        
        batch_sizes = [100, 500, 1000, 10000]        
        batch_sizes = [50]
        
        testcase = "sql"
        
        for batch_size in batch_sizes:
            options['batch_size'] = batch_size
            # Select method
            sqn = SQN(options)
            #sqn = PSQN(options)
            f = lambda b: print_f_vals(testcase, rowlim, options, folderpath, sqn)
            print(f(batch_size))
#            p = Process(target=f, args=(batch_size,))
#            p.start()
