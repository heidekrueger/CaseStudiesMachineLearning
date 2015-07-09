"""
@author: heidekrueger
Performs analysis and plotting on outputs of SQN benchmarking run
"""

import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn.datasets import load_iris as load_iris
from SQN.LogisticRegression import LogisticRegression as LR

def get_batchsizes_from_name(filepath):
    filename = filepath.split("/")[-1]
    filename = filename.split("_")
    b_G = int(filename[0])
    b_H = int(filename[1])
    return b_G, b_H

def get_filepaths(b_G, b_H, upd_stp = 1):
    """
    Returns the corresponding result paths for a pair of batch sizes
    """
    filedir = '../outputs/iris/'
    filename = str(b_G) + '_' + str(b_H) + '_' + str(upd_stp) + '.txt'
    filename_w = filename + '_w.txt'
    return filedir+filename, filedir+filename_w

def get_fixed_sample(size):
    """Returns X, y consisting of the first size rows of the dataset"""
    logreg = LR()
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X[:size], y[:size]

def get_fixed_F(size):
    """
    Returns the obj. function limited to the first size rows of dataset:
    F = 1/size * sum_i f(w, x_i, y_i) 
    """
    X_fix, y_fix = get_fixed_sample(size)
    return lambda w: logreg.F(w,X_fix,y_fix)

def load_result_file(filepath):
    
    resfile = open(filepath, "r")
    
    iters, fevals, gevals, adp, f_S, g_norm_S, time = [], [], [], [], [], [], []
    for count, line in enumerate(iter(resfile)):
        line = re.sub('\s', '', str(line))
        entries = re.split(",", str(line))
        
        iters.append( int(entries[0]))
        fevals.append( int(entries[1]) )
        gevals.append( int(entries[2]) )
        adp.append( int(entries[3]) )
        f_S.append( float(entries[4]) )
        g_norm_S.append( float(entries[5]) )
        time.append( float(entries[6]) )
        
    return iters, fevals, gevals, adp, f_S, g_norm_S, time
        
def load_result_file_w(filepath):
    
    resfile = open(filepath, "r")
    w = []
    for count, line in enumerate(iter(resfile)):
        line = re.sub('\s', '', str(line))
        entries = re.split(",", str(line))
        w.append(np.array([float(s) for s in entries]))
    return w

def get_moving_average(values, memory_length):
    return [np.mean(values[:i]) if i<=memory_length else np.mean(values[i-memory_length:i]) for i in range(len(values))]

"""
########################
Here be the action:
########################


Init Params:
"""
maxIters = 200
fixed_F_size = 1000

# will consider corresponding pairs of these:
b_G = [10, 10, 50, 50, 50, 150]
b_H = [0, 10, 0, 10, 50, 0]

#make color cycle

color_cycle=iter(cm.gist_rainbow(np.linspace(0,1,len(b_G))))


logreg = LR()
F = get_fixed_F(fixed_F_size)

""" Initialize plots: """
stochF_vs_iters = plt.figure(1)
plt.title("Sample Objective vs. Iterations")
plt.ylabel("Stochastic objective on batch")
plt.xlabel("Iterations")


stochF_vs_time = plt.figure(2)
plt.title("Sample Objective vs. CPU time (s)")

stochF_vs_adp = plt.figure(3)
plt.title("Sample Objective vs. Accessed Data Points")
#plt.xscale('log')

stochF_vs_fevals = plt.figure(4)
plt.title("Sample Objective vs. Function Evaluations")
#plt.xscale('log')



fixed_vs_iters = plt.figure(5)
plt.title("Fixed Subset Objective vs. Iterations")

fixed_vs_time = plt.figure(6)
plt.title("Fixed Subset Objective vs. CPU time (s)")

fixed_vs_adp = plt.figure(7)
plt.title("Fixed Subset Objective vs. Accessed Data Points")
#plt.xscale('log')

fixed_vs_fevals = plt.figure(8)
plt.title("Fixed Subset Objective vs. Function Evaluations")
#plt.xscale('log')

for i in range(8):
    plt.figure(i+1)
    plt.yscale('log')


for bg, bh in zip(b_G, b_H):
    """Load the results """
    filepath, filepath_w = get_filepaths(bg, bh)
    iters, fevals, gevals, adp, f_S, g_norm_S, time = load_result_file(filepath)
    print filepath_w
    w = load_result_file_w(filepath_w)

    """Plot the results """ 
    # next color
    c = next(color_cycle)

    if bh == 0:
        l = 'SGD, b: '+str(bg)
        ls = '--'
    elif bg <1000:
        l = 'SQN, bG '+str(bg)+' bH '+str(bh)
        ls = '-'
    else:
        l = 'SQN, bG '+str(bg/1000)+'k bH '+str(bh)
        ls = '-'

    plt.figure(1)
    plt.plot(iters[:maxIters], f_S[:maxIters], label = l, c=c, ls=ls)
    # plot moving averages
    #plt.plot(iters[:maxIters], get_moving_average(f_S,100)[:maxIters], label = ('Avg bG '+str(bg)+' bH '+str(bh)))

    plt.figure(2)
    plt.plot(time[:maxIters], f_S[:maxIters], label = l, c=c, ls=ls)

    plt.figure(3)
    plt.plot(adp[:maxIters], f_S[:maxIters], label = l, c=c, ls=ls)

    plt.figure(4)
    plt.plot(fevals[:maxIters], f_S[:maxIters], label = l, c=c, ls=ls)

    
    

    # get vals on fixed set
    print str(bg), str(bh)
    print len(fevals), len(w)
    Fvals = [F(w_i) for w_i in w[1:maxIters+1]]
    print len(iters), len(Fvals)
    plt.figure(5)
    plt.plot(iters[:maxIters], Fvals, label = l, c=c, ls=ls)

    plt.figure(6)
    plt.plot(time[:maxIters], Fvals, label = l, c=c, ls=ls)

    plt.figure(7)
    plt.plot(adp[:maxIters], Fvals, label = l, c=c, ls= ls)

    plt.figure(8)
    plt.plot(fevals[:maxIters], Fvals, label = l, c=c, ls=ls)

for i in range(8):
    plt.figure(i+1)
    plt.legend()


plt.show()







# 

# X_f, y_f = get_fixed_sample(5)

# logreg.w=w[-1]
# yp = logreg.predict(X_f)

# print yp, y_f

#v = [F(w_i) for w_i in w]

#fig = plt.figure()
#plt.plot(iters, v)
#plt.yscale('log')
#plt.show()

